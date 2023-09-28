package com.example.CSTRL.cst.behavior.RL.RLRates;

public class FlatRate extends RLRate {
    public FlatRate(Double a) {
        super(a);
    }

    @Override
    public Double getRate() {
        return a;
    }

    @Override
    public String toString() {
        return "FlatRate(a=" + a.toString() + ")";
    }
}
