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
}
