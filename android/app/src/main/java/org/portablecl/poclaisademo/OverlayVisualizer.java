package org.portablecl.poclaisademo;

import static android.graphics.Color.rgb;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
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

    Random rand;
    Paint paint;

    public OverlayVisualizer(){

        rand = new Random( 42);

        paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(3);
        paint.setTextSize(30);
        paint.setTextAlign(Paint.Align.LEFT);

    }

    /**
     * draw the detected objects on to the surfaceview provided
     * @param detections
     * @param surfaceView
     */
    public void DrawOverlay(int detections[], SurfaceView surfaceView){

        int numDetections = detections[0];

       int width =  surfaceView.getWidth();
       int height = surfaceView.getWidth();

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

           int color = rgb(rand.nextInt(155) +100,rand.nextInt(155) +100,
                   rand.nextInt(155) +100);
           paint.setColor(color);

           canvas.drawRect(box_x, box_y,box_x + box_w, box_y + box_h, paint);
           canvas.drawText(overlayLabel, box_x, box_y -30, paint);
           
       }

       holder.unlockCanvasAndPost(canvas);

       return;

    }
}
