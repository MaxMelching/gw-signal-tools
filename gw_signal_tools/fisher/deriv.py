import numpy as np


class Derivative():
    def __init__(
        self,
        wf_params_at_point,
        param_to_vary,
        wf_generator,
        step_size
    ) -> None:
        assert param_to_vary in wf_params_at_point

        self._wf_params = wf_params_at_point
        self._param_to_vary = param_to_vary
        self._wf_generator = wf_generator
        self._step_size = step_size

        self._wf = wf_generator(wf_params_at_point)

    @property
    def wf_params(self):
        return self._wf_params

    @property
    def param_to_vary(self):
        return self._param_to_vary

    @property
    def wf_generator(self):
        return self._wf_generator
    
    @property
    def step_size(self):
        return self._step_size
    
    @property
    def param_center_val(self):
        return self.wf_params[self.param_to_vary]

    @property
    def wf(self):
        return self._wf

    @property
    def wf_deriv(self):
        return 

    def wf_amplitude(self, wf):
        return np.abs(wf)
    
    @property
    def amplitude(self):
        return self.wf_amplitude(self.wf)

    @property
    def amplitude_deriv(self):
        pass

    def wf_phase(self, wf):
        return np.unwrap(np.angle(wf))
    
    @property
    def phase(self):
        return self.wf_phase(self.wf)

    @property
    def phase_deriv(self):
        pass
