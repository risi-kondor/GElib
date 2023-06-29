import torch
import gelib
torch.manual_seed(123)

batch_size = 200
maxl = 5
device = 'cpu'

signals = gelib.SO3vec.Frandn(batch_size, maxl, device)

## TESTS FAILS, ERROR MESSAGE:
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/runner.py:341: in from_call
#     result: Optional[TResult] = func()
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/runner.py:372: in <lambda>
#     call = CallInfo.from_call(lambda: list(collector.collect()), "collect")
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/python.py:531: in collect
#     self._inject_setup_module_fixture()
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/python.py:545: in _inject_setup_module_fixture
#     self.obj, ("setUpModule", "setup_module")
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/python.py:310: in obj
#     self._obj = obj = self._getobj()
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/python.py:528: in _getobj
#     return self._importtestmodule()
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/python.py:617: in _importtestmodule
#     mod = import_path(self.path, mode=importmode, root=self.config.rootpath)
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/pathlib.py:565: in import_path
#     importlib.import_module(module_name)
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/importlib/__init__.py:126: in import_module
#     return _bootstrap._gcd_import(name[level:], package, level)
# <frozen importlib._bootstrap>:1050: in _gcd_import
#     ???
# <frozen importlib._bootstrap>:1027: in _find_and_load
#     ???
# <frozen importlib._bootstrap>:1006: in _find_and_load_unlocked
#     ???
# <frozen importlib._bootstrap>:688: in _load_unlocked
#     ???
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/_pytest/assertion/rewrite.py:178: in exec_module
#     exec(co, module.__dict__)
# python/tests/manual_tests/test_FFT_accuracies.py:12: in <module>
#     inverse = gelib.SO3iFFT(signals, bandwidth)
# python/src/gelib/SO3vec.py:600: in SO3iFFT
#     return v.iFFT(N)
# python/src/gelib/SO3vec.py:246: in iFFT
#     return SO3vec_iFFTFn.apply(_N,*(self.parts))
# /opt/anaconda3/envs/gelib-test-ci/lib/python3.10/site-packages/torch/autograd/function.py:506: in apply
#     return super().apply(*args, **kwargs)  # type: ignore[misc]
# python/src/gelib/SO3vec.py:467: in forward
#     _r=ctensorb.view(r)
# E   RuntimeError: expected scalar type ComplexFloat


#for bandwidth in range(1, 101):
#    inverse = gelib.SO3iFFT(signals, bandwidth)
#    signals_ = gelib.SO3FFT(inverse, maxl)
#    sum_errors = 0
#    total_elems = 0
#    for l in range(len(signals.parts)):
#        error = torch.sum(torch.abs(torch.tensor(signals.parts[l]) - torch.tensor(signals_.parts[l])))
#        print(torch.tensor(error))
#        total_elems += signals.parts[l].numel()
#    average_error = sum_errors / total_elems
#    print(bandwidth, sum_errors, average_error)
