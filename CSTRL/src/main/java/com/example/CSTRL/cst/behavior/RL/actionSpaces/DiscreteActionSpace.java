package com.example.CSTRL.cst.behavior.RL.actionSpaces;

import java.util.ArrayList;
import java.util.Random;

public class DiscreteActionSpace extends ActionSpace {
    public DiscreteActionSpace(ArrayList<ArrayList<Double>> actions) {
        domain = actions;
    }

    public void setDomain(ArrayList<ArrayList<Double>> actions) {
        domain = actions;
    }

    @Override
    public ArrayList<Double> getRandomAction() {
        Random r = new Random();

        return domain.get(Math.abs(r.nextInt()) % domain.size());
    }

    @Override
    public ArrayList<Double> translateAPIAction(ArrayList<Double> a) {
        int n = (int) Math.round(a.get(0));

        if (n < 0) {
            System.out.println("Warning: given action " + n + " is less than 0");
            n = 0;
        } else if (n >= domain.size()) {
            System.out.println("Warning: given action " + n + " is greater than domain size");
            n = domain.size() - 1;
        }

        return domain.get(n);
    }
}
