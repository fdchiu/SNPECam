/*
 *  Created by David Chiu
 *  Dec. 28th, 2018
 *
 */
package com.davidchiu.snpecam;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import static com.davidchiu.snpecam.Detection.COLORS;

public class Ncnn
{
    public Vector<String> mLabel = new Vector<>();
    private int imageWidth;
    private int imageHeight;

    public native boolean init(byte[] param, byte[] bin, byte[] words);

    public native float[] nativeDetect(Bitmap bitmap);

    static {
        System.loadLibrary("ncnn_jni");
    }

    public void setImageSize(int width, int height) {
        imageHeight = height;
        imageWidth = width;
    }

    public boolean initNcnn(Context context, String paramFile, String weightsFile, String labels) throws IOException
    {
        byte[] param = null;
        byte[] bin = null;
        byte[] words = null;

        if (paramFile == null || weightsFile == null) {
            return false;
        }
        Log.i(Ncnn.class.getSimpleName(), "Loading: " + paramFile + " weight file: " + weightsFile);
        {
            InputStream assetsInputStream;
                assetsInputStream = context.getAssets().open(paramFile);
            int available = assetsInputStream.available();
            param = new byte[available];
            int byteCode = assetsInputStream.read(param);
            assetsInputStream.close();
        }
        {
            InputStream assetsInputStream = context.getAssets().open(weightsFile);
            int available = assetsInputStream.available();
            bin = new byte[available];
            int byteCode = assetsInputStream.read(bin);
            assetsInputStream.close();
        }
        if (labels != null)
        {
            InputStream assetsInputStream = context.getAssets().open(labels);
            int available = assetsInputStream.available();
            words = new byte[available];
            int byteCode = assetsInputStream.read(words);

            assetsInputStream.reset();
            BufferedReader br = null;
            br = new BufferedReader(new InputStreamReader(assetsInputStream));
            String line;
            while ((line = br.readLine()) != null) {
                mLabel.add(line);
            }
            br.close();

            assetsInputStream.close();
        }

        return init(param, bin, words);
    }

    //
    // This needs to be customized based on underling models: yolo, ssd etc.
    // Now only yolo is supported
    //
    public List<Detection> detect(Bitmap image) {
        float[] result = nativeDetect(image);
        List<Detection> list=null;
        if (result == null || result[0] <= 0) return list;

        list = parseNcnnResults(result);

        return list;
    }

    List<Detection> parseNcnnResults(float[] ncnnArray) {
        ArrayList<Detection> list = new ArrayList<>();
        if (ncnnArray == null || ncnnArray.length<=0 || ncnnArray[0]<=0) {
            return  null;
        }

        int nItems = (int)((ncnnArray.length-1)/ncnnArray[0]);
        int nParams = (int)ncnnArray[0];
        for (int i = 0; i < nItems; i++) {
            String id = "" + (int)(ncnnArray[i*nParams+1]);
            RectF location = new RectF(ncnnArray[i*nParams+3]*imageWidth, ncnnArray[i*nParams+4]*imageHeight, ncnnArray[i*nParams+5]*imageWidth, ncnnArray[i*nParams+6]*imageHeight);
            Detection recognition = new Detection();
            recognition.id = (int)ncnnArray[i*nParams + 1];
            recognition.title = mLabel.get(recognition.id);
            recognition.detectionConfidence = ncnnArray[i*nParams+2];
            recognition.location = location;
            recognition.color = COLORS[0];
            list.add(recognition);
            //Log.i("ncnn", "detect: " + ncnnArray[i*nParams+1] + " " +  ncnnArray[i*nParams+2] + " " +
            //       ncnnArray[i*nParams+3]  + " " + ncnnArray[i*nParams+4] + " " +  ncnnArray[i*nParams+5] + " " + ncnnArray[i*nParams+6]);
        }
        return list;
    }

}
