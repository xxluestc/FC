"""Post-execution health-observation loop around the mechanistic world model."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fc_power.health.observer import (
    DegradationObservation,
    HealthBelief,
    HealthObserver,
    HealthObserverUpdate,
)
from fc_power.world_model.mechanistic import (
    MechanisticMultiStackWorldModel,
    MultiStackAction,
    MultiStackState,
    StackControlState,
    WorldStep,
)


@dataclass(frozen=True)
class ObservedMultiStackState:
    """World-model state plus one explicit observer belief per stack."""

    model_state: MultiStackState
    beliefs: tuple[HealthBelief, ...]

    def __post_init__(self) -> None:
        if len(self.beliefs) != len(self.model_state.stacks):
            raise ValueError("one health belief is required per stack")
        for index, (stack, belief) in enumerate(
            zip(self.model_state.stacks, self.beliefs)
        ):
            if stack.health != belief.state:
                raise ValueError(
                    f"stack {index} model health and observer belief must match"
                )


@dataclass(frozen=True)
class StackObservedUpdate:
    stack_index: int
    update: HealthObserverUpdate


@dataclass(frozen=True)
class ObservedWorldStep:
    """Executed model prediction and the corrected state for the next decision."""

    prediction: WorldStep
    next_state: ObservedMultiStackState
    observers: tuple[StackObservedUpdate, ...]


class ObservedHealthExecutionLoop:
    """Apply observations only after an action has been selected and executed.

    Candidate allocators continue to call ``model.step`` directly.  This wrapper
    is intentionally named ``execute`` so a planner cannot silently consume an
    end-of-step observation during hypothetical rollouts.
    """

    def __init__(
        self,
        model: MechanisticMultiStackWorldModel,
        observers: tuple[HealthObserver, ...],
    ):
        if not isinstance(model, MechanisticMultiStackWorldModel):
            raise TypeError("model must be MechanisticMultiStackWorldModel")
        if len(observers) != model.n_stacks:
            raise ValueError("one health observer is required per stack")
        self.model = model
        self.observers = tuple(observers)

    def initialize(self, model_state: MultiStackState) -> ObservedMultiStackState:
        if not isinstance(model_state, MultiStackState):
            raise TypeError("model_state must be MultiStackState")
        if len(model_state.stacks) != self.model.n_stacks:
            raise ValueError("model_state stack count does not match the model")
        beliefs = tuple(
            observer.initialize(stack.health)
            for observer, stack in zip(self.observers, model_state.stacks)
        )
        return ObservedMultiStackState(model_state=model_state, beliefs=beliefs)

    def execute(
        self,
        state: ObservedMultiStackState,
        action: MultiStackAction,
        demand_power_kw: float,
        *,
        observations: tuple[DegradationObservation | None, ...] | None = None,
        allow_dwell_override: bool = False,
    ) -> ObservedWorldStep:
        """Predict with the conditional mean, then correct the next-step state."""

        if not isinstance(state, ObservedMultiStackState):
            raise TypeError("state must be ObservedMultiStackState")
        if len(state.beliefs) != self.model.n_stacks:
            raise ValueError("observer state stack count does not match the model")
        supplied = (
            tuple(None for _ in range(self.model.n_stacks))
            if observations is None
            else tuple(observations)
        )
        if len(supplied) != self.model.n_stacks:
            raise ValueError("observations must match the stack count")

        prediction = self.model.step(
            state.model_state,
            action,
            demand_power_kw,
            stochastic_health=False,
            allow_dwell_override=allow_dwell_override,
        )
        posterior_stacks: list[StackControlState] = []
        posterior_beliefs: list[HealthBelief] = []
        observer_updates: list[StackObservedUpdate] = []
        for index, (observer, prior, predicted_stack, stack_step, observation) in enumerate(
            zip(
                self.observers,
                state.beliefs,
                prediction.next_state.stacks,
                prediction.stacks,
                supplied,
            )
        ):
            observer_prediction = observer.predict(
                prior,
                predicted_stack.health,
                expected_gamma_increment_pct=(
                    stack_step.expected_load_increment_pct
                ),
            )
            if observation is None:
                update = HealthObserverUpdate(
                    observer_prediction, observer_prediction, None
                )
            else:
                posterior, audit = observer.correct(
                    observer_prediction,
                    observation,
                    monotonic_lower_bound_pct=prior.state.degradation,
                )
                update = HealthObserverUpdate(
                    observer_prediction, posterior, audit
                )
            posterior_stacks.append(
                replace(predicted_stack, health=update.posterior.state)
            )
            posterior_beliefs.append(update.posterior)
            observer_updates.append(StackObservedUpdate(index, update))

        posterior_model_state = MultiStackState(
            soc=prediction.next_state.soc,
            stacks=tuple(posterior_stacks),
            elapsed_s=prediction.next_state.elapsed_s,
        )
        next_state = ObservedMultiStackState(
            model_state=posterior_model_state,
            beliefs=tuple(posterior_beliefs),
        )
        return ObservedWorldStep(
            prediction=prediction,
            next_state=next_state,
            observers=tuple(observer_updates),
        )
