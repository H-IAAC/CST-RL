package com.example.CSTRL.cst;

import br.unicamp.cst.core.entities.Codelet;
import br.unicamp.cst.core.entities.MemoryObject;
import br.unicamp.cst.core.entities.Mind;
import com.example.CSTRL.cst.motor.MotorCodelet;
import com.example.CSTRL.cst.perception.PerceptionCodelet;
import com.example.CSTRL.util.RLPercept;

import java.util.ArrayList;
import java.util.Arrays;

abstract public class AgentMind extends Mind {

    private MemoryObject stateMO;
    private MemoryObject actionMO;

    public AgentMind() {
        /*
            INITIALIZE MEMORY OBJECTS
        */

        // Sensor
        stateMO = createMemoryObject("STATE", new RLPercept(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.0, false));

        // Behavior
        MemoryObject RLPerceptMO = createMemoryObject("RLPERCEPT", new RLPercept(new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.0, false));
        MemoryObject RLActionMO = createMemoryObject("RLACTION", new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0)));

        // Motor
        actionMO = createMemoryObject("ACTION", new ArrayList<Double>(Arrays.asList(0.0, 0.0, 0.0, 0.0)));

        /*
            INITIALIZE CODELETS
        */

        // Perception
        PerceptionCodelet perceptionCodelet = new PerceptionCodelet();

        perceptionCodelet.addInput(stateMO);
        perceptionCodelet.addOutput(RLPerceptMO);

        perceptionCodelet.setIsMemoryObserver(true);
        stateMO.addMemoryObserver(perceptionCodelet);

        insertCodelet(perceptionCodelet, "PERCEPTION");

        // Behavior
        Codelet RLCodelet = getRLCodelet(RLPerceptMO);

        RLCodelet.addInput(RLPerceptMO);
        RLCodelet.addOutput(RLActionMO);

        insertCodelet(RLCodelet, "BEHAVIOR");

        // Motor
        MotorCodelet motorCodelet = new MotorCodelet();

        motorCodelet.addInput(RLActionMO);
        motorCodelet.addOutput(actionMO);

        motorCodelet.setIsMemoryObserver(true);
        RLActionMO.addMemoryObserver(motorCodelet);

        insertCodelet(motorCodelet, "MOTOR");

        /*
            INITIALIZE MIND
        */

        // Define timing
        for (Codelet c : getCodeRack().getAllCodelets()) {
            c.setTimeStep(8);
        }

        start();
    }

    public void setState(RLPercept percept) {
        stateMO.setI(percept);
    }

    public ArrayList<Double> getAction() {
        return (ArrayList<Double>) actionMO.getI();
    }

    abstract protected Codelet getRLCodelet(MemoryObject perceptMO);
}
