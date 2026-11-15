"""Independent policy conditions with OR gating and one shared behavior injection."""


class PolicyCAST:
    def __init__(self, model, vectors, conditions, alpha):
        from safetune.steer import CASTModel
        import math
        import torch
        if not conditions:
            raise ValueError('At least one policy condition is required')
        for condition in conditions.values():
            if (not torch.isfinite(condition.condition_vector).all()
                    or condition.condition_vector.norm() == 0 or not math.isfinite(condition.threshold)):
                raise ValueError('Nonfinite or degenerate policy condition')
        if not vectors or any(not torch.isfinite(v).all() for v in vectors.values()):
            raise ValueError('Missing or nonfinite behavior vectors')
        self.model = model
        self.gates = {name: CASTModel(model, steering_vectors={}, condition=condition,
                                     alpha=alpha, per_prompt_gate=True)
                      for name, condition in conditions.items()}
        self.behavior = CASTModel(model, steering_vectors=vectors,
            condition=next(iter(conditions.values())), alpha=alpha, per_prompt_gate=True)
        self.active_policies = ()
        self.last_gate = {}

    def set_active_policies(self, policies):
        if not policies or len(set(policies)) != len(policies):
            raise ValueError('Active policies must be nonempty and unique')
        missing = set(policies) - self.gates.keys()
        if missing:
            raise ValueError(f'Uncalibrated policies: {sorted(missing)}')
        self.active_policies = tuple(policies)

    def _gate_fires(self, input_ids, **kwargs):
        if not self.active_policies:
            raise ValueError('Set active policies before generation')
        fires = [False] * input_ids.shape[0]
        self.last_gate = {}
        for name in self.active_policies:
            gate = self.gates[name]
            decisions, scores = gate._gate_fires(input_ids, **kwargs)
            if not isinstance(decisions, list) or len(decisions) != len(fires):
                raise RuntimeError('Policy gate must return one decision per prompt')
            self.last_gate[name] = {'fires': decisions, 'scores': scores,
                'layer': gate.condition_layer, 'threshold': gate.threshold,
                'comparator': gate.comparator}
            fires = [a or b for a, b in zip(fires, decisions)]
        # Different condition scores/thresholds are not a common numerical scale.
        scores = (self.last_gate[self.active_policies[0]]['scores']
                  if len(self.active_policies) == 1 else [None] * len(fires))
        return fires, scores

    def generate(self, *args, **kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        if ids is None:
            raise ValueError('PolicyCAST requires input_ids')
        gate_kwargs = {'attention_mask': kwargs['attention_mask']} if 'attention_mask' in kwargs else {}
        if ids.shape[0] > 1 and 'attention_mask' in kwargs:
            # Forward (unlike generate) does not derive positions from left padding.
            positions = kwargs['attention_mask'].long().cumsum(-1) - 1
            gate_kwargs['position_ids'] = positions.masked_fill(kwargs['attention_mask'] == 0, 0)
        fires, _ = self._gate_fires(ids, **gate_kwargs)
        # SafeTune handles the per-row mask and removes hooks in a finally block.
        return self.behavior._steered(fires, self.model.generate, *args, **kwargs)
