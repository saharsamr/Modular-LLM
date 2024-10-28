import numpy as np
import torch
import tqdm


def compute_loglike_loss(logits, labels, reduction="none"):
    bs = logits.size(0)
    vocab_size = logits.size(-1)
    labels = labels.squeeze(-1)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    # reshape back
    if reduction == "none":
        loss = loss.view((bs, -1))
        # mean only non-zero
        non_zero_loss = (loss != 0).sum(dim=-1)
        non_zero_loss[non_zero_loss == 0] = 1
        loss = loss.sum(dim=-1) / non_zero_loss
    return loss


class Evaluator(ABC):
    def __init__(
        self,
        datamodule=None,
        config=None,
        use_vllm=False,
        **_,
    ):
        if config is None and datamodule is None:
            raise ValueError("Either config or datamodule must be provided.")

        self.datamodule = datamodule
        if config is None:
            config = datamodule.config

        self.config = deepcopy(config)
        self.use_vllm = use_vllm
        self._last_metrics = None

    def get_dataloader(self, split, subsample, shuffle):
        if self.datamodule is None:
            raise ValueError("No datamodule initialized!")

        if split in ["test", "testing"]:
            dataloader = self.datamodule.test_dataloader(subsample, shuffle)
        elif split in ["train", "training"]:
            dataloader = self.datamodule.train_dataloader(subsample)
        elif split in ["val", "valid", "validation", "dev"]:
            dataloader = self.datamodule.val_dataloader(subsample, shuffle)
        else:
            raise ValueError("Unknown split: {}".format(split))
        return dataloader

    @property
    def last_metrics(self):
        return self._last_metrics

    def save_metrics(self, metrics, output_path, predictions=None):
        import json

        class JsonCustomEncoder(json.JSONEncoder):
            """<cropped for brevity>"""

            def default(self, obj):
                if isinstance(obj, (np.ndarray, np.number)):
                    return obj.tolist()
                elif isinstance(obj, set):
                    return list(obj)
                elif isinstance(obj, bytes):  # pragma: py3
                    return obj.decode()
                return json.JSONEncoder.default(self, obj)

        self._last_metrics = metrics

        if output_path is None:
            return

        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

        with open(output_path + "/metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, cls=JsonCustomEncoder)

        if predictions is not None:
            with open(output_path + "/predictions.json", "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)

    @abstractmethod
    def evaluate(
        self,
        model,
        split=None,
        shuffle=False,
        subsample=-1,
        output_path=None,
        **kwargs,
    ):
        pass

    def evaluate_with_vllm(self, model, dataloader, num_batches=None, verbose=True):
        raise NotImplementedError()

    @property
    def tasks(self):
        self.datamodule.task_names

    @property
    def tokenizer(self):
        return self.datamodule.tokenizer


class LogLikeEvaluator(Evaluator):
    def __init__(self, datamodule, **kwargs):
        super().__init__(datamodule=datamodule, **kwargs)

    @switch_to_eval_mode
    def evaluate(
        self,
        model,
        split="val",
        subsample=-1,
        num_batches=None,
        verbose=True,
        shuffle=False,
        output_path=None,
    ):
        from mttl.models.expert_model import BaseExpertModel
        from mttl.models.lightning.base_module import LightningEfficientCheckpoint
        from mttl.models.utils import transfer_batch_to_device

        dataloader = self.get_dataloader(split, subsample, shuffle=shuffle)

        if self.use_vllm:
            return self.evaluate_with_vllm(model, dataloader, num_batches, verbose)

        pbar = tqdm.tqdm(
            enumerate(dataloader),
            total=len(dataloader),
        )

        all_losses = []
        all_accuracies = []
        all_predictions = []

        device = next(model.parameters()).device

        for num_batch, batch in pbar:
            if num_batches is not None and num_batch >= num_batches:
                break

            labels_index = batch.pop("labels_index", None)
            num_options = batch.pop("num_options")
            labels_texts = batch.pop("labels_texts")
            sources_texts = batch.pop("sources_texts")
            batch_size = len(labels_index)

            batch = transfer_batch_to_device(batch, device)

            with torch.no_grad():
                if isinstance(model, LightningEfficientCheckpoint) or isinstance(
                    model, BaseExpertModel
                ):
                    logits = model.forward(**batch).logits
                else:
                    logits = model.forward(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    ).logits

                loss_per_option = compute_loglike_loss(
                    logits, batch["labels"], reduction="none"
                )
                loss_per_option = loss_per_option.cpu().numpy()
                loss_per_example = [
                    loss_per_option[
                        int(np.sum(num_options[:i])) : int(np.sum(num_options[: i + 1]))
                    ]
                    for i in range(batch_size)
                ]
                predictions = [
                    np.argmin(option_loss) for option_loss in loss_per_example
                ]

                all_predictions.extend(predictions)
                all_losses.extend(loss_per_option.tolist())

                if labels_index is not None:
                    all_accuracies.extend(
                        (np.array(predictions) == np.array(labels_index)).tolist()
                    )

            if verbose:
                logger.info("Sources:\n%s", sources_texts[0])
                logger.info("Label:\n%s", labels_texts[labels_index[0]])
                logger.info("Prediction:\n%s", labels_texts[predictions[0]])

            if all_accuracies:
                pbar.set_description("Accuracy: {:.4f}".format(np.mean(all_accuracies)))

        metrics = {
            "loss": float(np.mean(all_losses)),
            "loglike": -float(np.mean(all_losses)),
            "predictions": all_predictions,
            "accuracy": float(np.mean(all_accuracies)) if all_accuracies else None,
        }

        self.save_metrics(metrics, output_path)
        return metrics["accuracy"]
