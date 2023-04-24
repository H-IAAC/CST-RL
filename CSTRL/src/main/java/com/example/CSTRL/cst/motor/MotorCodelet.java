package com.example.CSTRL.cst.motor;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;

import java.util.ArrayList;

public class MotorCodelet extends Codelet {
    private MemoryObject RLActionMO;
    private MemoryObject actionMO;

    @Override
    public void accessMemoryObjects() {
        RLActionMO = (MemoryObject) getInput("RLACTION");
        actionMO = (MemoryObject) getOutput("ACTION");
    }

    @Override
    public void calculateActivation() {

    }

    @Override
    public void proc() {
        ArrayList<Double> actionList = (ArrayList<Double>) RLActionMO.getI();

        for (int i = 0; i < actionList.size(); i++) {
            Double value = actionList.get(i);

            if (value > 1.0) {
                value = 1.0;
            } else if (value < 0.0) {
                value = 0.0;
            }

            actionList.set(i, value);
        }

        actionMO.setI(actionList);
    }
}
