package com.example.CSTRL.util;

public class RLAction extends GodotContainer {
    private final Object action;

    public RLAction(Object action) {
        super(ReturnType.ACTION);
        this.action = action;
    }

    public Object getAction() {
        return action;
    }
}
