package com.example.CSTRL;

import com.example.CSTRL.cst.*;
import com.example.CSTRL.util.*;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;

@RestController
public class Controller {
    private AgentMind agentMind;

    // Initialize
    @GetMapping("/initialize")
    public GodotContainer initialize() {
        agentMind = new QLearningAgentMind();

        return new GodotContainer(ReturnType.INIT);
    }

    // Updates StateMO, runs up to action update, then returns action
    @PostMapping("/step")
    public GodotContainer step(@RequestBody RLPercept percept) {
        if (agentMind == null) {
            initialize();
        }

        agentMind.setState(percept);

        if (percept.getEnded()) {
            return new GodotContainer(ReturnType.RESET);
        }

        return new RLAction(agentMind.getAction());
    }

    // Test get mapping
    @GetMapping("/gettest")
    public TestContainer getTest(@RequestParam(defaultValue = "NAN") String t) {
        return new TestContainer("API is running!", t);
    }

    // Test post mapping
    @PostMapping("/posttest")
    public TestContainer postTest(@RequestBody TestContainer testContainer) {
        return new TestContainer("API is running! - " + testContainer.getMessage(), testContainer.getAddon());
    }

    // Type test
    @PostMapping("/typetest")
    public TypeTestContainer typeTest(@RequestBody TypeTestContainer typeTestContainer) {
        return new TypeTestContainer("API is running! - " + typeTestContainer.getMessage() + " - " + typeTestContainer.getAddon(), typeTestContainer.getArray().toString() + " - " + Integer.toString(typeTestContainer.getAnInt()), typeTestContainer.getArray(), typeTestContainer.getAnInt());
    }
}
