package com.example.CSTRL.cst.behavior.RL.actionSpaces;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class ContinuousActionSpace extends ActionSpace {
    // Here, domain will be an ArrayList with size n where n is the degree of freedom of the action space. Each element
    // of the domain will be an ArrayList with 2 elements: the minimum and maximum value of that dimension, respectively

    final Double DEFAULT_MIN = 0.0;
    final Double DEFAULT_MAX = 1.0;

    public ContinuousActionSpace(int size) {
        domain = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            appendToDomain(DEFAULT_MIN, DEFAULT_MAX);
        }
    }

    public void setDomain(ArrayList<Double> minValues, ArrayList<Double> maxValues) {
        domain = new ArrayList<>();

        for (int i = 0; i < Math.max(minValues.size(), maxValues.size()); i++) {
            Double min = i >= minValues.size() ? DEFAULT_MIN : minValues.get(i);
            Double max = i >= maxValues.size() ? DEFAULT_MAX : maxValues.get(i);

            appendToDomain(min, max);
        }
    }

    private void appendToDomain(Double min, Double max) {
        ArrayList<Double> a = new ArrayList<>();
        a.add(DEFAULT_MIN);
        a.add(DEFAULT_MAX);

        domain.add(a);
    }

    @Override
    public ArrayList<Double> getRandomAction() {
        Random r = new Random();
        ArrayList<Double> action = new ArrayList<>();

        for (int i = 0; i < domain.size(); i++) {
            Double min = domain.get(i).get(0);
            Double max = domain.get(i).get(1);
            action.add(r.nextDouble() * (max - min) + min);
        }

        return action;
    }
}
