package com.example.CSTRL.cst.behavior;

import br.unicamp.cst.core.entities.MemoryObject;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.actionSelectors.ActionSelector;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public class StateActionValueFunctionRLCodelet extends EpisodicRLCodelet {

    Policy policy;
    ActionSelector actionSelector;
    StateActionValueFunction stateActionValueFunction;

    public StateActionValueFunctionRLCodelet(MemoryObject perceptMO, Policy policy, ActionSelector actionSelector, StateActionValueFunction stateActionValueFunction) {
        super(perceptMO);

        this.policy = policy;
        this.actionSelector = actionSelector;
        this.stateActionValueFunction = stateActionValueFunction;
    }

    @Override
    protected void runRLStep() {
        stateActionValueFunction.update(pastState, pastAction, currentState, reward, policy);
    }

    @Override
    protected ArrayList<Double> selectAction() {
        return actionSelector.selectAction(currentState, stateActionValueFunction, policy);
    }

    @Override
    protected void newEpisode() {
        super.newEpisode();

        policy.endEpisode();
        actionSelector.endEpisode();
        stateActionValueFunction.endEpisode();
    }
}
