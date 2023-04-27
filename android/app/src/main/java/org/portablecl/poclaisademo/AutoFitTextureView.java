package org.portablecl.poclaisademo;

import android.content.Context;
import android.util.AttributeSet;
import android.view.TextureView;

import androidx.annotation.NonNull;

/**
 *  a TextureView that keeps the aspect ratio set to it, even after resizing
 */
public class AutoFitTextureView extends TextureView {

    private int ratioWidth = 0;
    private int ratioHeight = 0;

    public AutoFitTextureView(@NonNull Context context) {
        this(context, null);
    }

    public AutoFitTextureView(Context context, AttributeSet attrs) {
        this(context, attrs, 0);
    }

    public AutoFitTextureView(Context context, AttributeSet attrs, int defStyle) {
        super(context, attrs, defStyle);
    }

    /**
     * set the aspect ratio wanted for the texture
     * the values are relative, so 640:480 is 4:3
     *
     * @param width
     * @param height
     */
    public void setAspectRatio(int width, int height) {
        assert width > 0;
        assert height > 0;

        ratioWidth = width;
        ratioHeight = height;

        requestLayout();
    }

    @Override
    protected void onMeasure(int widthMeaSpec, int heightMeasureSpec) {
        super.onMeasure(widthMeaSpec, heightMeasureSpec);

        int newWidth = MeasureSpec.getSize(widthMeaSpec);
        int newHeight = MeasureSpec.getSize(heightMeasureSpec);

        if (ratioWidth == 0 || ratioHeight == 0) {
            setMeasuredDimension(newWidth, newHeight);
        } else {
            // if the aspect ratio is not the same, set new height according to new width
            if (newWidth < (newHeight * ratioWidth) / ratioHeight) {
                setMeasuredDimension(newWidth, (newWidth * ratioHeight) / ratioWidth);
            } else {
                setMeasuredDimension((newHeight * ratioWidth) / ratioHeight, newHeight);
            }
        }

    }
}
