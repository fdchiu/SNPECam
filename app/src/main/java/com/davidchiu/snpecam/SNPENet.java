package com.davidchiu.snpecam;

import android.app.Activity;
import android.app.Application;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.Tensor;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Vector;

import static com.davidchiu.snpecam.Detection.COLORS;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.DSP;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.GPU;
import static com.qualcomm.qti.snpe.NeuralNetwork.Runtime.CPU;

public class SNPENet {

    public final String LOG_TAG = this.getClass().getSimpleName();
    public static final int MAX_RESULTS = 10;
    public static final float CONFIDENCE_THRESHOLD= 0.5f;

    private String modelFile;
    private String labelFile;
    public Vector<String> mLabel = new Vector<>();
    private int imageWidth = 640;
    private int imageHeight = 480;

    private NeuralNetwork mNNetwork=null;
    private NeuralNetwork.Runtime mTargetRuntime=CPU;
    private List<NeuralNetwork.Runtime> supportedRuntime = new LinkedList<>();

    private final SupportedTensorFormat mTensorFormat = SupportedTensorFormat.FLOAT;
    private String  modelName = "ssdmobilenet";
    String mInputLayer=null;
    String mOutputLayer=null;
    long mJavaExecuteTime = -1;
    public enum SupportedTensorFormat {
        FLOAT,
        UB_TF8
    }

    public void setImageSize(int width, int height) {
        imageHeight = height;
        imageWidth = width;
    }

    /**
     * Interface methods
     */

    /**
     *
     * @return: String for the name of running target
     */
    public String getRunningTargegt() {
        if (mTargetRuntime == DSP) return "DSP";
        else if (mTargetRuntime == GPU) return "GPU";
        else return "CPU";
    }
    public void setModelName(String name) { modelName = name;}
    public void init(Activity activity, String modelFile, String labelFile){
        try {
            loadLabels(activity.getApplicationContext(), labelFile);
        } catch (IOException e) {
            Log.e(this.getClass().getSimpleName(), " " + e);
        }

        //Application app=context.getApplicationContext();

        int modelSize=0;
        InputStream assetsInputStream = null;
        try {
            assetsInputStream = activity.getAssets().open(modelFile);
            modelSize = assetsInputStream.available();
        } catch ( IOException e) {
            Log.e(this.getClass().getSimpleName(), " Failed open model: " + e);

        }

        try {
            final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(activity.getApplication())
                    .setDebugEnabled(true)
                    .setModel(assetsInputStream, modelSize)
                    .setCpuFallbackEnabled(true)
                    .setUseUserSuppliedBuffers(mTensorFormat != SupportedTensorFormat.FLOAT);

            for (NeuralNetwork.Runtime runtime : NeuralNetwork.Runtime.values()) {
                if (builder.isRuntimeSupported(runtime)) {
                    supportedRuntime.add(runtime);
                }
            }
            mTargetRuntime = getPreferedRuntime();
            builder.setRuntimeOrder(mTargetRuntime);
            if (modelName.equals("ssdmobilenet")) {
                builder.setOutputLayers("Postprocessor/BatchMultiClassNonMaxSuppression", "add_6");
            } else if(modelName.equals("caffessd")) {
                builder.setOutputLayers("detection_out");
            }
            final long start = SystemClock.elapsedRealtime();
            mNNetwork = builder.build();
            final long end = SystemClock.elapsedRealtime();

            //mLoadTime = end - start;
        } catch (IllegalStateException | IOException e) {
            Log.e(LOG_TAG, e.getMessage(), e);
        }

        Set<String> inputNames = mNNetwork.getInputTensorsNames();
        Set<String> outputNames = mNNetwork.getOutputTensorsNames();
        if (modelName.equals("ssdmobilenet")) {
            mInputLayer = inputNames.iterator().next();
            mOutputLayer = outputNames.iterator().next();

        } else {
            if (inputNames.size() != 1 || outputNames.size() != 1) {
                throw new IllegalStateException("Invalid network input and/or output tensors.");
            } else {
                mInputLayer = inputNames.iterator().next();
                mOutputLayer = outputNames.iterator().next();
                Log.i(this.getClass().getSimpleName(), "inputNames: " + inputNames.toString());
                Log.i(this.getClass().getSimpleName(), "outputNames: " + outputNames.toString());
            }
        }

    }

    public void destroy() {
        if (mNNetwork != null) {
            mNNetwork.release();
            mNNetwork = null;
        }
    }

    public List<Detection> run(Bitmap bitmap) {
        final FloatTensor tensor = mNNetwork.createFloatTensor(
                mNNetwork.getInputTensorsShapes().get(mInputLayer));

        //loadMeanImageIfAvailable(mModel.meanImage, tensor.getSize());

        final int[] dimensions = tensor.getShape();
        final boolean isGrayScale = (dimensions[dimensions.length -1] == 1);
        float[] rgbBitmapAsFloat;
        if (!isGrayScale) {
            rgbBitmapAsFloat = loadRgbBitmapAsFloat(bitmap);
        } else {
            rgbBitmapAsFloat = loadGrayScaleBitmapAsFloat(bitmap);
        }
        tensor.write(rgbBitmapAsFloat, 0, rgbBitmapAsFloat.length);

        final Map<String, FloatTensor> inputs = new HashMap<>();
        inputs.put(mInputLayer, tensor);

        final long javaExecuteStart = SystemClock.elapsedRealtime();
        final Map<String, FloatTensor> outputs = mNNetwork.execute(inputs);
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

        float[] results;
        float[] classes = new float[MAX_RESULTS];
        float[] scores = new float[MAX_RESULTS];
        float[] bboxes = new float[MAX_RESULTS*4];
        int nSize = MAX_RESULTS*6;
        results = new float[nSize+1];
        results[0] = 6;//nSize;
        int nClass=0;
        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            String key = output.getKey();
            if (output.getKey().equals(mOutputLayer)) {
            //if (key.startsWith("Postprocessor/BatchMultiClassNonMaxSuppression")) {
                FloatTensor outputTensor = output.getValue();

                final float[] array = new float[outputTensor.getSize()];
                outputTensor.read(array, 0, array.length);
                int n1 = array.length;
                int n2 = outputTensor.getSize();
                if (n2 >MAX_RESULTS) {
                    n2=MAX_RESULTS;
                }
                if (key.endsWith("classes")) {
                    System.arraycopy(array, 0, classes, 0, MAX_RESULTS);
                } else if (key.endsWith("scores")) {
                    System.arraycopy(array, 0, scores, 0, MAX_RESULTS);
                } else if (key.endsWith("boxes")) {
                    System.arraycopy(array, 0, bboxes, 0, MAX_RESULTS*4);
                }
                String outStr= key + " count: " + n1 ;
                for (int i=0; i< n2; i++) {
                    outStr += " " + array[i];
                }
                Log.i("run", outStr);
                /*for (Pair<Integer, Float> pair : topK(1, array)) {
                    result.add(mModel.labels[pair.first]);
                    result.add(String.valueOf(pair.second));
                }*/
            }
        }
        for (int i=0; i< MAX_RESULTS; i++ ) {
            if (scores[i] > CONFIDENCE_THRESHOLD) {
                results[i * 6 + 1] = classes[i];
                results[i * 6 + 2] = scores[i];
                System.arraycopy(bboxes, i * 4, results, i * 6 + 3, 4);
            }
        }
        releaseTensors(inputs, outputs);
        return parseResults(results);
    }


    private void loadLabels(Context context, String labelsFile) throws IOException {

        InputStream assetsInputStream = context.getAssets().open(labelsFile);
        int available = assetsInputStream.available();
        BufferedReader br = null;
        br = new BufferedReader(new InputStreamReader(assetsInputStream));
        String line;
        while ((line = br.readLine()) != null) {
            mLabel.add(line);
        }
        br.close();

        assetsInputStream.close();

    }

    float[] loadRgbBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length * 3];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;
                final int batchIdx = idx * 3;

                final float[] rgb = extractColorChannels(pixels[idx]);
                pixelsBatched[batchIdx]     = rgb[0];
                pixelsBatched[batchIdx + 1] = rgb[1];
                pixelsBatched[batchIdx + 2] = rgb[2];
            }
        }
        return pixelsBatched;
    }

    float[] loadGrayScaleBitmapAsFloat(Bitmap image) {
        final int[] pixels = new int[image.getWidth() * image.getHeight()];
        image.getPixels(pixels, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight());

        final float[] pixelsBatched = new float[pixels.length];
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                final int idx = y * image.getWidth() + x;

                final int rgb = pixels[idx];
                final float b = ((rgb)       & 0xFF);
                final float g = ((rgb >>  8) & 0xFF);
                final float r = ((rgb >> 16) & 0xFF);
                float grayscale = (float) (r * 0.3 + g * 0.59 + b * 0.11);

                pixelsBatched[idx] = preProcess(grayscale);
            }
        }
        return pixelsBatched;
    }

    private float[] extractColorChannels(int pixel) {
        String modelName = this.modelName;
        FloatBuffer mMeanImage = null;

        float b = ((pixel)       & 0xFF);
        float g = ((pixel >>  8) & 0xFF);
        float r = ((pixel >> 16) & 0xFF);

        if (modelName.equals("inception_v3")) {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return new float[] {preProcess(b), preProcess(g), preProcess(r)};
        } else {
            return new float[] {preProcess(r), preProcess(g), preProcess(b)};
        }
    }

    private float preProcess(float original) {
        String modelName = this.modelName;
        FloatBuffer mMeanImage = null;

        if (modelName.equals("inception_v3") || modelName.equals("ssdmobilenet")) {
            return (original - 128) / 128;
        } else if (modelName.equals("alexnet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else if (modelName.equals("googlenet") && mMeanImage != null) {
            return original - mMeanImage.get();
        } else {
            return original;
        }
    }

    Pair<Integer, Float>[] topK(int k, final float[] tensor) {
        final boolean[] selected = new boolean[tensor.length];
        final Pair<Integer, Float> topK[] = new Pair[k];
        int count = 0;
        while (count < k) {
            final int index = top(tensor, selected);
            selected[index] = true;
            topK[count] = new Pair<>(index, tensor[index]);
            count++;
        }
        return topK;
    }

    private int top(final float[] array, boolean[] selected) {
        int index = 0;
        float max = -1.f;
        for (int i = 0; i < array.length; i++) {
            if (selected[i]) {
                continue;
            }
            if (array[i] > max) {
                max = array[i];
                index = i;
            }
        }
        return index;
    }


    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }

    List<Detection> parseResults(float[] ncnnArray) {
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
            Log.i("snpe", "detect: " + ncnnArray[i*nParams+1] + " " +  ncnnArray[i*nParams+2] + " " +
                   ncnnArray[i*nParams+3]  + " " + ncnnArray[i*nParams+4] + " " +  ncnnArray[i*nParams+5] + " " + ncnnArray[i*nParams+6]);
        }
        return list;
    }

    public List<NeuralNetwork.Runtime> getSupportedRuntimeList() {
        return supportedRuntime;
    }

    private NeuralNetwork.Runtime getPreferedRuntime() {
        if (supportedRuntime == null || supportedRuntime.size()<=0) {
            supportedRuntime.add(CPU);
            return CPU;
        }
        NeuralNetwork.Runtime selected = CPU;
        for (NeuralNetwork.Runtime runtime: supportedRuntime) {
            if (runtime == DSP) {
                return DSP;
            } else {
                if (runtime == GPU) {
                    selected = GPU;
                }
            }
        }
        return selected;
        //return CPU;
    }
}
