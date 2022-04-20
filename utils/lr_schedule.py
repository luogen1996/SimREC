class LR_Scheduler:

    def __init__(self, schedule_func, tensorboard, init_epoch=0, verbose=0):
        super(LR_Scheduler, self).__init__()
        self.schedule_func = schedule_func
        self.verbose = verbose
        self.epoch = init_epoch
        self.lr = 0.

    def __call__(self, epoch, optimizer):
        self.epoch += 1
        self.lr = self.schedule_func(self.epoch)
        for param_group in optimizer.param_groups():
            param_group['lr'] = self.lr
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %.4f' % (self.epoch, self.lr))
