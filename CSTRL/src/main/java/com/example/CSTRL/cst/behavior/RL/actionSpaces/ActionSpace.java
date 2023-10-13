package com.example.CSTRL.cst.behavior.RL.actionSpaces;

import java.util.ArrayList;

abstract public class ActionSpace {
    protected ArrayList<ArrayList<Double>> domain;

    public ArrayList<ArrayList<Double>> getDomain() {return domain;}

    abstract public ArrayList<Double> getRandomAction();

    abstract public ArrayList<Double> translateAPIAction(ArrayList<Double> a);
}
