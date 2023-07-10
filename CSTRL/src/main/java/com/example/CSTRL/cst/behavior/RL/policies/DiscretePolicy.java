package com.example.CSTRL.cst.behavior.RL.policies;

import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public class DiscretePolicy extends Policy {
    ArrayList<ArrayList<Double>> actionList;

    public DiscretePolicy(ArrayList<ArrayList<Double>> actionList) {
        this.actionList = actionList;
    }

    @Override
    public ArrayList<Double> getPolicyAction(ArrayList<Double> S, StateActionValueFunction valueFunction) {
        ArrayList<Double> bestAction = new ArrayList<Double>();
        Double bestValue = Double.NEGATIVE_INFINITY;

        for (ArrayList<Double> action : actionList) {
            Double value = valueFunction.getValue(S, action);
            if (value > bestValue) {
                bestValue = value;
                bestAction = action;
            }
        }

        return bestAction;
    }

    @Override
    public ArrayList<Double> getRandomAction() {
        Random r = new Random();
        return actionList.get(r.nextInt(actionList.size()));
    }
}
