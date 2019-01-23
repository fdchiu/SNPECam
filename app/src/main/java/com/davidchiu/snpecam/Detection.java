package com.davidchiu.snpecam;

import android.graphics.Color;
import android.graphics.RectF;

public class Detection {
    int id;
    RectF location;
    float detectionConfidence;
    int color;
    String title;

    public static final int[] COLORS = {
            Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.CYAN, Color.MAGENTA, Color.WHITE,
            Color.parseColor("#55FF55"), Color.parseColor("#FFA500"), Color.parseColor("#FF8888"),
            Color.parseColor("#AAAAFF"), Color.parseColor("#FFFFAA"), Color.parseColor("#55AAAA"),
            Color.parseColor("#AA33AA"), Color.parseColor("#0D0068")
    };

}
