package com.example.CSTRL.cst.behavior.RL.actionManagers;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public abstract class ActionManager extends RLElement {
    public abstract ArrayList<Double> getBestAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction);

    public abstract ArrayList<Double> getRandomAction();
}
