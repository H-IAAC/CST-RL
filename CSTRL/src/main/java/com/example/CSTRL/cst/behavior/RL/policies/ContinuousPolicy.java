package com.example.CSTRL.cst.behavior.RL.policies;

import com.example.CSTRL.cst.behavior.RL.RLRates.LinearDecreaseRLRate;
import com.example.CSTRL.cst.behavior.RL.RLRates.RLRate;
import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public abstract class ContinuousPolicy extends Policy {
    protected final int totalActions;
    protected ArrayList<Double> actionMinimums;
    protected ArrayList<Double> actionMaximums;
    protected RLRate alpha;

    public ContinuousPolicy(int totalActions, RLRate alpha) {
        this.totalActions = totalActions;
        this.alpha = alpha;

        actionMinimums = new ArrayList<>();
        actionMaximums = new ArrayList<>();
        for (int i = 0; i < totalActions; i++) {
            actionMinimums.add(0.0);
            actionMaximums.add(1.0);
        }
    }

    abstract public void updatePolicy(ArrayList<Double> initialState);

    public void setActionMinimums(ArrayList<Double> actionMinimums) {
        if (actionMinimums.size() == totalActions) {
            this.actionMinimums = actionMinimums;
        }
    }

    public void setActionMaximums(ArrayList<Double> actionMaximums) {
        if (actionMaximums.size() == totalActions) {
            this.actionMaximums = actionMaximums;
        }
    }

    @Override
    public ArrayList<Double> getRandomAction() {
        ArrayList<Double> action = new ArrayList<>();
        Random r = new Random();

        for (int i = 0; i < totalActions; i++) {
            action.add(actionMinimums.get(i) + r.nextDouble() * (actionMaximums.get(i) - actionMinimums.get(i)));
        }

        return action;
    }

    abstract public ArrayList<Double> getBestAction(ArrayList<Double> S);
}
