from collections import OrderedDict, defaultdict
from functools import partial

import torch
import torch.nn as nn
from torch.nn import Parameter as P
from pretorched.models.torchvision_models import resnet18


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = pretorched.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = pretorched.models.utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class Normalize(nn.Module):
    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        shape=(1, -1, 1, 1, 1),
        rescale=True,
    ):
        super().__init__()
        self.shape = shape
        self.mean = P(torch.tensor(mean).view(shape), requires_grad=False)
        self.std = P(torch.tensor(std).view(shape), requires_grad=False)
        self.rescale = rescale

    def forward(self, x, rescale=None):
        rescale = self.rescale if rescale is None else rescale
        x.div_(255.0) if rescale else None
        return (x - self.mean) / self.std


class FeatureHooks:
    def __init__(self, hooks, named_modules):
        # setup feature hooks
        modules = {k: v for k, v in named_modules}
        for h in hooks:
            hook_name = h['name']
            m = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            if h['type'] == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif h['type'] == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, name, *args):
        x = args[
            -1
        ]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][name] = x

    def get_output(self, device):
        output = tuple(self._feature_outputs[device].values())[::-1]
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output


# model = resnet18()
# nm = model.named_modules()
# hooks = [{'name': 'layer1.1.conv1', 'type': 'forward'}]
# f = FeatureHooks(hooks, nm)
# x = torch.randn(4, 3, 224, 224)
# out = model(x)
# o = f.get_output(torch.device('cpu'))
# print(o[0].shape)
