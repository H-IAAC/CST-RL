package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.policies.Policy;
import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;

import java.util.ArrayList;

public abstract class StateActionValueFunction extends RLElement {
    RLRate explorationRate;
    Double discountRate;

    public StateActionValueFunction(RLRate explorationRate, Double discountRate) {
        this.explorationRate = explorationRate;
        this.discountRate = discountRate;
    }

    public abstract Double getValue(ArrayList<Double> state, ArrayList<Double> action);

    public abstract void update(ArrayList<Double> pastState, ArrayList<Double> pastAction, ArrayList<Double> state, Double reward, Policy policy);

    // Given an action vector, returns the sub-array of the gradient elements obtained through the
    // differentiation with relation to these actions
    public abstract ArrayList<Double> getActionGradient(ArrayList<Double> S, ArrayList<Double> A);

    @Override
    public void endEpisode() {
        explorationRate.endEpisode();
    }
}
