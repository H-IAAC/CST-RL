package com.example.CSTRL.cst.behavior.RL.actionSelectors;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public abstract class ActionSelector extends RLElement {

    public abstract ArrayList<Double> selectAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, Policy policy);

    protected ArrayList<Double> selectOptimalAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, Policy policy) {
        return policy.getPolicyAction(state, stateActionValueFunction);
    }
}
