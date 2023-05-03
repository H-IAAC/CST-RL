package com.example.CSTRL.cst.behavior.RL.valueFunctions;

import com.example.CSTRL.cst.behavior.RL.RLElement;
import com.example.CSTRL.cst.behavior.RL.actionManagers.ActionManager;
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

    public abstract void update(ArrayList<Double> pastState, ArrayList<Double> pastAction, ArrayList<Double> state, Double reward, ActionManager actionManager);

    @Override
    public void endEpisode() {
        explorationRate.endEpisode();
    }
}
