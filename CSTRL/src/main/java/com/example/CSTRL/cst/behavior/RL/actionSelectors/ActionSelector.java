package com.example.CSTRL.cst.behavior.RL.actionSelectors;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public abstract class ActionSelector extends RLElement {

    public abstract ArrayList<Double> selectAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, ActionManager actionManager);

    protected ArrayList<Double> selectOptimalAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction, ActionManager actionManager) {
        return actionManager.getBestAction(state, stateActionValueFunction);
    }
}
