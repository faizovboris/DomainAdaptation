import os


def simple_callback(model, epoch_log, current_epoch, total_epoch):
    train_loss = epoch_log['loss']
    val_loss = epoch_log['val_loss']
    trg_metrics = epoch_log['trg_metrics']
    src_metrics = epoch_log['src_metrics']
    message_head = f'Epoch {current_epoch+1}/{total_epoch}\n'
    message_loss = 'loss: {:<10}\t val_loss: {:<10}\t'.format(train_loss, val_loss)
    message_src_metrics = ' '.join(['val_src_{}: {:<10}\t'.format(k, v) for k, v in src_metrics.items()])
    message_trg_metrics = ' '.join(['val_trg_{}: {:<10}\t'.format(k, v) for k, v in trg_metrics.items()])
    print(message_head + message_loss + message_src_metrics + message_trg_metrics)


class ModelSaver:
    def __init__(self, model_type, save_freq=1, path="checkpoints"):
        self.model_type = model_type
        self.path = path
        self.save_freq = save_freq
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(os.path.join(path, model_type)):
            os.makedirs(os.path.join(path, model_type))

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        import torch
        if current_epoch % self.save_freq == 0:
            filename = os.path.join(self.path, self.model_type, "epoch_{}.pt".format(current_epoch))
            torch.save(model.state_dict(), filename)


class HistorySaver:
    import json
    from collections import defaultdict
    json = json
    defaultdict = defaultdict

    def __init__(self, log_name, val_freq=1, path="_log", plot=True):
        self.log_name = log_name
        self.path = path
        self.is_plotting = plot
        self.val_freq = val_freq
        self.loss_history = self.defaultdict(list)
        self.src_metrics_history = self.defaultdict(list)
        self.trg_metrics_history = self.defaultdict(list)

        if plot:
            import matplotlib.pyplot as plt
            self.plt = plt
        if not os.path.exists(path):
            os.makedirs(path)

    def _plot(self, data, name, current_epoch, total_epoch):
        self.plt.figure(figsize=(6, 4))

        for key in data:
            if key != 'loss':
                self.plt.plot(list(range(0, current_epoch + 1, self.val_freq)), data[key], label=key)
            else:
                self.plt.plot(data[key], label=key)

        self.plt.legend()
        self.plt.title('{} history for {} epochs of {}'.format(name, current_epoch + 1, total_epoch))
        self.plt.savefig(os.path.join(self.path, name +'_plot'))

    def plot_all(self, current_epoch, total_epoch):
        self._plot(self.loss_history, 'loss', current_epoch, total_epoch)
        self._plot(self.src_metrics_history, 'src_metrics', current_epoch, total_epoch)
        self._plot(self.trg_metrics_history, 'trg_metrics', current_epoch, total_epoch)
        self.plt.close('all')

    def _save_to_json(self, data, name=None):
        filename = os.path.join(self.path, name)
        with open(filename, 'w') as f:
            self.json.dump(data, f)

    def __call__(self, model, epoch_log, current_epoch, total_epoch):
        if current_epoch % self.val_freq == 0:
            self.loss_history['val_loss'].append(epoch_log['val_loss'])

            for metric in epoch_log['trg_metrics']:
                self.trg_metrics_history[metric].append(epoch_log['trg_metrics'][metric])

            for metric in epoch_log['src_metrics']:
                self.src_metrics_history[metric].append(epoch_log['src_metrics'][metric])

        self.loss_history['loss'].append(epoch_log['loss'])

        self._save_to_json(self.loss_history, name='loss_history')
        self._save_to_json(self.src_metrics_history, name='src_metrics')
        self._save_to_json(self.trg_metrics_history, name='trg_metrics')

        if self.is_plotting:
            self.plot_all(current_epoch, total_epoch)