package com.example.CSTRL.util;

import java.util.ArrayList;

public class RLPercept {
    private final ArrayList<Double> state;
    private ArrayList<Double> action;
    private final Double reward;
    private final boolean terminal;

    public RLPercept(ArrayList<Double> state, Double reward, boolean terminal) {
        this.state = state;
        this.reward = reward;
        this.terminal = terminal;
    }

    public void setAction(ArrayList<Double> action) {
        this.action = action;
    }

    public ArrayList<Double> getState() {
        return state;
    }

    public ArrayList<Double> getAction() {
        return action;
    }

    public Double getReward() {
        return reward;
    }

    public boolean isTerminal() {
        return terminal;
    }
}
