package org.portablecl.poclaisademo;

import static android.graphics.Color.rgb;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

import java.util.Random;

/**
 * a class that draws overlay received from the pocl image processor
 */
public class OverlayVisualizer {

    private String classes[] =
            {"person",        "bicycle",      "car",
            "motorcycle",    "airplane",     "bus",
            "train",         "truck",        "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench",        "bird",
            "cat",           "dog",          "horse",
            "sheep",         "cow",          "elephant",
            "bear",          "zebra",        "giraffe",
            "backpack",      "umbrella",     "handbag",
            "tie",           "suitcase",     "frisbee",
            "skis",          "snowboard",    "sports ball",
            "kite",          "baseball bat", "baseball glove",
            "skateboard",    "surfboard",    "tennis racket",
            "bottle",        "wine glass",   "cup",
            "fork",          "knife",        "spoon",
            "bowl",          "banana",       "apple",
            "sandwich",      "orange",       "broccoli",
            "carrot",        "hot dog",      "pizza",
            "donut",         "cake",         "chair",
            "couch",         "potted plant", "bed",
            "dining table",  "toilet",       "tv",
            "laptop",        "mouse",        "remote",
            "keyboard",      "cell phone",   "microwave",
            "oven",          "toaster",      "sink",
            "refrigerator",  "book",         "clock",
            "vase",          "scissors",     "teddy bear",
            "hair drier",    "toothbrush"};

    private int colors[] =
            {-1651865,-6634562,-5921894,
            -9968734,-1277957,-2838283,
            -9013359,-9634954,-470042,
            -8997255,-4620585,-2953862,
            -3811878,-8603498,-2455171,
            -5325920,-6757258,-8214427,
            -5903423,-4680978,-4146958,
            -602947,-5396049,-9898511,
            -8346466,-2122577,-2304523,
            -4667802,-222837,-4983945,
            -234790,-8865559,-4660525,
            -3744578,-8720427,-9778035,
            -680538,-7942224,-7162754,
            -2986121,-8795194,-2772629,
            -4820488,-9401960,-3443339,
            -1781041,-4494168,-3167240,
            -7629631,-6685500,-6901785,
            -2968136,-3953703,-4545430,
            -6558846,-2631687,-5011272,
            -4983118,-9804322,-2593374,
            -8473686,-4006938,-7801488,
            -7161859,-4854121,-5654350,
            -817410,-8013957,-9252928,
            -2240041,-3625560,-6381719,
            -4674608,-5704237,-8466309,
            -1788449,-7283030,-5781889,
            -4207444,-8225948};

    private Paint paint;
    private float fontSize;

    public OverlayVisualizer(){

        fontSize = 30;
        paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3);
        paint.setTextSize(fontSize);
        paint.setTextAlign(Paint.Align.LEFT);

    }

    /**
     * draw the detected objects on to the surfaceview provided
     * @param detections
     * @param surfaceView
     */
    public void DrawOverlay(int detections[], Size captureSize,
                            boolean rotated, SurfaceView surfaceView){

        int numDetections = detections[0];

       int width =  surfaceView.getWidth();
       int height = surfaceView.getHeight();

        float xScale, yScale;
       if(rotated){
            xScale = (float) surfaceView.getWidth()/ captureSize.getHeight();
            yScale = (float) surfaceView.getHeight()/ captureSize.getWidth();
       }else {
           xScale = (float) surfaceView.getWidth()/ captureSize.getWidth();
           yScale = (float) surfaceView.getHeight()/ captureSize.getHeight();
       }


       float confidence;
       int classIds;
       String className;
       String overlayLabel;
       int box_x, box_y, box_w, box_h;

       SurfaceHolder holder = surfaceView.getHolder();
       holder.setFormat(PixelFormat.TRANSPARENT);
       Canvas canvas = holder.lockCanvas();

       canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

       for(int i = 0; i < numDetections; i++){

           classIds = detections[1 + 6 * i];
           className = classes[classIds];
           confidence = Float.intBitsToFloat(detections[1 + 6 * i + 1]);
           overlayLabel = className + String.format(" %.2f", confidence);

           box_x = detections[1 + 6 * i + 2];
           box_y = detections[1 + 6 * i + 3];
           box_w = detections[1 + 6 * i + 4];
           box_h = detections[1 + 6 * i + 5];

           paint.setColor(colors[classIds]);

           canvas.drawRect(box_x * xScale, box_y * yScale,(box_x + box_w) * xScale,
                   (box_y + box_h) * yScale, paint);
           canvas.drawText(overlayLabel, box_x * xScale, (box_y * yScale) -fontSize, paint);

       }

       holder.unlockCanvasAndPost(canvas);

       return;

    }
}
