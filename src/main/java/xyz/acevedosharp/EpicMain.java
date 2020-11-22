package xyz.acevedosharp;

import java.util.Collections;

public class EpicMain {
    // dataset, minutes, repetitions, cores, runHasco, runMLPlan
    public static void main(String[] args) {
        System.out.println("Max mem: " + Runtime.getRuntime().maxMemory());

        EpicExecutor epicExecutor = new EpicExecutor(
                Collections.singletonList(args[0]),
                Integer.parseInt(args[1]),
                Integer.parseInt(args[2]),
                Integer.parseInt(args[3])
        );

        boolean runHasco = Boolean.parseBoolean(args[4]);
        boolean runMLPlan = Boolean.parseBoolean(args[5]);

        try {
            if (runHasco && runMLPlan) {
                epicExecutor.runBoth();
            } else if (runHasco) {
                epicExecutor.callHascoRunner();
            } else {
                epicExecutor.callMLPlanRunner();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
