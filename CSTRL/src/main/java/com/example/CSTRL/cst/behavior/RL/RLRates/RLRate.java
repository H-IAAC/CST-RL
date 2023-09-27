package com.example.CSTRL.cst.behavior.RL.RLRates;

import com.example.CSTRL.cst.behavior.RL.RLElement;

public abstract class RLRate extends RLElement {
    Double a;

    public RLRate(Double a) {
        this.a = a;
    }

    public abstract Double getRate();

    public void endEpisode() {
        return;
    }
}
