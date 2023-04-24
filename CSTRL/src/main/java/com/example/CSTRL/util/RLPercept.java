package com.example.CSTRL.util;

import java.util.ArrayList;

public class RLPercept {
    private final ArrayList<Double> state;
    private final Double reward;
    private final boolean ended;

    public RLPercept(ArrayList<Double> state, Double reward, boolean ended) {
        this.state = state;
        this.reward = reward;
        this.ended = ended;
    }

    public ArrayList<Double> getState() {
        return state;
    }

    public Double getReward() {
        return reward;
    }

    public boolean getEnded() {
        return ended;
    }
}
