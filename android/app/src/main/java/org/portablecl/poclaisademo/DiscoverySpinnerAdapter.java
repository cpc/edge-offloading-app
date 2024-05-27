package org.portablecl.poclaisademo;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import androidx.annotation.NonNull;

import java.util.List;

/*
 * This class extends ArrayAdapter to modify the spinner view.
 */
public class DiscoverySpinnerAdapter extends ArrayAdapter<DiscoverySelect.spinnerObject> {
    private Context context;
    private List<DiscoverySelect.spinnerObject> objects;

    public DiscoverySpinnerAdapter(@NonNull Context context, int resource,
                                   @NonNull List<DiscoverySelect.spinnerObject> objects) {
        super(context, resource, objects);
        this.context = context;
        this.objects = objects;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        TextView label = (TextView) super.getView(position, convertView, parent);
        if (position == 0) {
            label.setText(objects.get(position).getAddress());
            return label;
        }
        objects.get(position).updatePing();
        label.setText(objects.get(position).getDescription());

        return label;
    }

    @Override
    public View getDropDownView(int position, View convertView, ViewGroup parent) {
        TextView label = (TextView) super.getDropDownView(position, convertView, parent);
        for (DiscoverySelect.spinnerObject s : objects) {
            if(s.address.equals(DiscoverySelect.DEFAULT_SPINNER_VAL)) {
                continue;
            }
            s.pingMonitor.tick();
        }
        if (position == 0) {
            label.setText(objects.get(position).getAddress());
            return label;
        }
        objects.get(position).updatePing();
        label.setText(objects.get(position).getDescription());

        return label;
    }
}
