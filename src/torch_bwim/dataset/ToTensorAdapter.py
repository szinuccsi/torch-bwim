class ToTensorAdapter(object):

    def __init__(self, num_of_features_out):
        super().__init__()
        self._num_of_features_out = num_of_features_out

    def _get_num_of_features_out(self):
        return self._num_of_features_out
    num_of_features_out = property(_get_num_of_features_out)

    def __call__(self, data: dict) -> tuple:
        pass
