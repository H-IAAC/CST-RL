package com.example.CSTRL.cst.behavior.RL.actionManagers;

import com.example.CSTRL.cst.behavior.RL.valueFunctions.StateActionValueFunction;

import java.util.ArrayList;
import java.util.Random;

public class DiscreteActionManager extends ActionManager {
    ArrayList<ArrayList<Double>> actionList;

    public DiscreteActionManager(ArrayList<ArrayList<Double>> actionList) {
        this.actionList = actionList;
    }

    @Override
    public ArrayList<Double> getBestAction(ArrayList<Double> state, StateActionValueFunction stateActionValueFunction) {
        ArrayList<Double> bestAction = new ArrayList<Double>();
        Double bestValue = Double.NEGATIVE_INFINITY;

        for (ArrayList<Double> action : actionList) {
            Double value = stateActionValueFunction.getValue(state, action);
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
