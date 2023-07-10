package com.example.CSTRL.cst.behavior.RL.policies;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;

public abstract class Policy extends RLElement {
    public abstract ArrayList<Double> getPolicyAction(ArrayList<Double> S, StateActionValueFunction valueFunction);

    public abstract ArrayList<Double> getRandomAction();
}
