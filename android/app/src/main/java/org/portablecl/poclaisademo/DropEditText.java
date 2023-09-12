package org.portablecl.poclaisademo;

import android.content.Context;
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.widget.EditText;

public class DropEditText extends androidx.appcompat.widget.AppCompatEditText {
    public DropEditText(Context context) {
        super(context);
    }

    public DropEditText(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public DropEditText(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    /**
     * this function loses focus of the edittext when the back button is pressed.
     * @param keyCode The value in event.getKeyCode().
     * @param event Description of the key event.
     * @return
     */
    @Override
    public boolean onKeyPreIme(int keyCode, KeyEvent event) {
        if (event.KEYCODE_BACK == keyCode && event.ACTION_UP == event.getAction()) {
            this.clearFocus();
        }
        return super.onKeyPreIme(keyCode, event);
    }
}
