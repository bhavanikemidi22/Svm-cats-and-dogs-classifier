import React, { useState, useEffect, useRef } from 'react';
import {
  Brain,
  Sliders,
  Play,
  RotateCcw,
  Upload,
  Plus,
  RefreshCw,
  Sparkles,
  Award,
  Activity,
  CheckCircle,
  FileImage,
  Layers,
  Search,
  Check,
  AlertCircle,
  Info,
  ChevronRight,
  TrendingUp,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { CAT_DOG_DATASET, DatasetItem } from './utils/dataset';
import { SVM, SVMConfig, SVMModel, KernelType } from './utils/svm';
import InteractiveCanvas from './components/InteractiveCanvas';
import FeatureScanner from './components/FeatureScanner';
import TheorySection from './components/TheorySection';

export default function App() {
  // --- STATE MANAGERS ---
  const [dataset, setDataset] = useState<DatasetItem[]>(CAT_DOG_DATASET);
  const [selectedPoint, setSelectedPoint] = useState<DatasetItem | null>(CAT_DOG_DATASET[0]);
  const [hoveredPoint, setHoveredPoint] = useState<DatasetItem | null>(null);

  // SVM Hyperparameters
  const [kernelType, setKernelType] = useState<KernelType>('rbf');
  const [C, setC] = useState<number>(3.0);
  const [rbfSigma, setRbfSigma] = useState<number>(0.35);
  const [polyDegree, setPolyDegree] = useState<number>(3);
  const [showMargins, setShowMargins] = useState<boolean>(true);

  // Training state
  const [trainingStatus, setTrainingStatus] = useState<'untrained' | 'training' | 'converged' | 'max_passes'>('untrained');
  const [currentPass, setCurrentPass] = useState<number>(0);
  const [totalAlphasChanged, setTotalAlphasChanged] = useState<number>(0);
  const [alphasChangedInLastPass, setAlphasChangedInLastPass] = useState<number>(0);

  // Active SVM Class instance as a persistent ref
  const svmRef = useRef<SVM | null>(null);
  const [svmModel, setSvmModel] = useState<SVMModel | null>(null);

  // Custom upload state
  const [uploading, setUploading] = useState<boolean>(false);
  const [showLabelModal, setShowLabelModal] = useState<{ url: string; warmth: number; edgeDensity: number } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Loop execution timer for live training animations
  const trainingTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize or reconfigure the SVM ref when hyperparameters are adjusted
  useEffect(() => {
    // Instantiate or reconfigure
    const config: Partial<SVMConfig> = {
      kernelType,
      C,
      rbfSigma,
      polyDegree,
      tolerance: 0.001,
      maxPasses: 10,
    };

    if (!svmRef.current) {
      svmRef.current = new SVM(config);
    } else {
      svmRef.current.setConfig(config);
    }

    // Changing hyperparameters resets model calculations to force retraining
    handleResetModel();
  }, [kernelType, C, rbfSigma, polyDegree]);

  // Clears active training runs
  const stopTrainingLoop = () => {
    if (trainingTimerRef.current) {
      clearInterval(trainingTimerRef.current);
      trainingTimerRef.current = null;
    }
  };

  // Resets the SVM state
  const handleResetModel = () => {
    stopTrainingLoop();
    if (svmRef.current) {
      svmRef.current.reset();
    }
    setSvmModel(null);
    setTrainingStatus('untrained');
    setCurrentPass(0);
    setTotalAlphasChanged(0);
    setAlphasChangedInLastPass(0);
  };

  // Run a SINGLE step pass of Platt's SMO optimization
  const handleStepTrain = (): boolean => {
    if (!svmRef.current) return false;

    const X = dataset.map((p) => p.features);
    const Y = dataset.map((p) => p.label);

    const result = svmRef.current.stepTrain(X, Y);
    const model = svmRef.current.getModel();
    setSvmModel(model);

    setAlphasChangedInLastPass(result.alphasChanged);
    setTotalAlphasChanged((prev) => prev + result.alphasChanged);

    return result.alphasChanged > 0;
  };

  // Run training loop step-by-step
  const handleAutoTrain = () => {
    handleResetModel();
    setTrainingStatus('training');

    const maxPasses = 50;
    let passesWithoutChange = 0;
    let pCount = 0;

    const runPass = () => {
      pCount++;
      const changed = handleStepTrain();

      if (changed) {
        passesWithoutChange = 0;
      } else {
        passesWithoutChange++;
      }

      setCurrentPass(pCount);

      const targetPassesWithoutChange = 8;
      if (passesWithoutChange >= targetPassesWithoutChange) {
        // Converged! No alphas changed for several passes
        stopTrainingLoop();
        setTrainingStatus('converged');
      } else if (pCount >= maxPasses) {
        stopTrainingLoop();
        setTrainingStatus('max_passes');
      }
    };

    // Run first pass instantly
    runPass();

    // Schedule remaining passes so user can watch the boundary morph and flex spectacularly!
    trainingTimerRef.current = setInterval(() => {
      runPass();
    }, 110);
  };

  // Handle addition of custom points with a modal
  const handleAddCustomPoint = (warmth: number, edgeDensity: number, type: 'cat' | 'dog') => {
    const id = `custom-${Date.now()}`;
    const newPoint: DatasetItem = {
      id,
      type,
      label: type === 'cat' ? 1 : -1,
      name: `Custom ${type === 'cat' ? 'Cat' : 'Dog'}`,
      breed: `User Coordinate Node`,
      imageUrl: type === 'cat' 
        ? 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200&q=80'
        : 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=200&q=80',
      features: [warmth, edgeDensity],
      description: `A custom user-defined dataset token situated at warmth coordinate ${warmth} and Sobel edge complexity ${edgeDensity}.`,
    };

    setDataset((prev) => [...prev, newPoint]);
    setSelectedPoint(newPoint);
    handleResetModel(); // Reset to fit new points
  };

  // Trigger file uploader selection
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Process custom uploaded file via FileReader and Canvas Sobel Extractor
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const reader = new FileReader();
    reader.onload = async (event) => {
      const dataUrl = event.target?.result as string;
      if (!dataUrl) {
        setUploading(false);
        return;
      }

      // Import feature extraction dynamically
      const { extractFeatures } = await import('./utils/featureExtractor');
      const features = await extractFeatures(dataUrl);

      setUploading(false);
      // Open label selection modal to ask if they want to add it as +1 (Cat) or -1 (Dog)
      setShowLabelModal({
        url: dataUrl,
        warmth: features.warmth,
        edgeDensity: features.edgeDensity,
      });
    };
    reader.readAsDataURL(file);
  };

  // Complete addition of custom uploaded image
  const handleConfirmCustomUpload = (type: 'cat' | 'dog') => {
    if (!showLabelModal) return;

    const id = `custom-upload-${Date.now()}`;
    const newPoint: DatasetItem = {
      id,
      type,
      label: type === 'cat' ? 1 : -1,
      name: `Uploaded ${type === 'cat' ? 'Cat' : 'Dog'}`,
      breed: 'Custom Upload',
      imageUrl: showLabelModal.url,
      features: [showLabelModal.warmth, showLabelModal.edgeDensity],
      description: 'An external user-uploaded image analyzed locally using offscreen Canvas and Sobel edge kernels.',
    };

    setDataset((prev) => [...prev, newPoint]);
    setSelectedPoint(newPoint);
    setShowLabelModal(null);
    handleResetModel(); // reset matrix to train on new data point
  };

  // Clear all custom data elements to return to standard Kaggle subset
  const handleResetDataset = () => {
    setDataset(CAT_DOG_DATASET);
    setSelectedPoint(CAT_DOG_DATASET[0]);
    setHoveredPoint(null);
    handleResetModel();
  };

  // Calculate diagnostic accuracy metrics against current dataset
  const calculateMetrics = () => {
    if (!svmModel || !svmRef.current) return { accuracy: 0, tp: 0, tn: 0, fp: 0, fn: 0, precision: 0, recall: 0, f1: 0, supportVectors: 0 };

    let correct = 0;
    let tp = 0; // True Cat
    let tn = 0; // True Dog
    let fp = 0; // Dog predicted as Cat (False Cat)
    let fn = 0; // Cat predicted as Dog (False Dog)

    dataset.forEach((p) => {
      const pred = svmRef.current!.predict(p.features);
      const actual = p.label;

      if (pred === actual) {
        correct++;
        if (actual === 1) tp++;
        else tn++;
      } else {
        if (actual === -1 && pred === 1) fp++;
        else fn++;
      }
    });

    const total = dataset.length;
    const accuracy = total > 0 ? (correct / total) * 100 : 0;
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    const supportVectors = svmModel.alphas.filter((a) => a > 1e-5).length;

    return {
      accuracy,
      tp,
      tn,
      fp,
      fn,
      precision,
      recall,
      f1,
      supportVectors,
    };
  };

  const metrics = calculateMetrics();

  // Highlight points details on Hover on plot
  const activeDetailPoint = hoveredPoint || selectedPoint;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 md:p-6 font-sans antialiased overflow-x-hidden selection:bg-indigo-500/30 selection:text-white">
      {/* 1. APP NAVBAR HEADER */}
      <header className="max-w-7xl mx-auto mb-6 flex flex-col md:flex-row md:items-center justify-between gap-4 bg-slate-900/60 border border-slate-800 p-5 rounded-2xl backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-tr from-amber-500/20 via-indigo-500/15 to-blue-500/20 text-indigo-400 rounded-xl border border-slate-700/50 shadow-inner">
            <Brain className="w-6 h-6 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-100 tracking-tight font-sans">
              SVM Vision Labs Classifier
            </h1>
            <p className="text-xs text-slate-400">
              Interactive Support Vector Machine classifying cats and dogs via Sobel texture maps & warmth coordinates
            </p>
          </div>
        </div>

        {/* Global Dataset Controls & Quick resets */}
        <div className="flex gap-2">
          <button
            onClick={handleResetDataset}
            className="px-3.5 py-1.5 bg-slate-950 hover:bg-slate-900 border border-slate-800 text-slate-300 text-xs font-mono rounded-lg transition-all flex items-center gap-1.5 cursor-pointer"
            title="Clear all uploaded and custom coordinate nodes"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset Dataset
          </button>
        </div>
      </header>

      {/* 2. DUAL LAYOUT: WORKBENCH GRID */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COMPARTMENT (COL SPAN 5): INTERACTIVE PLOT */}
        <section className="lg:col-span-5 flex flex-col gap-4">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
            <div className="flex justify-between items-center mb-3">
              <div className="flex items-center gap-1.5">
                <Activity className="w-4.5 h-4.5 text-indigo-400" />
                <h2 className="text-sm font-medium text-slate-200">2D Separation Space</h2>
              </div>
              <label className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono cursor-pointer">
                <input
                  type="checkbox"
                  checked={showMargins}
                  onChange={(e) => setShowMargins(e.target.checked)}
                  className="rounded bg-slate-950 border-slate-800 text-indigo-600 focus:ring-0 w-3.5 h-3.5"
                />
                Show Margins
              </label>
            </div>

            <InteractiveCanvas
              dataset={dataset}
              svmModel={svmModel}
              hoveredPoint={hoveredPoint}
              selectedPoint={selectedPoint}
              onSelectPoint={setSelectedPoint}
              onHoverPoint={setHoveredPoint}
              onAddCustomPoint={handleAddCustomPoint}
              showMargins={showMargins}
            />

            <p className="text-[10px] text-slate-500 font-mono mt-3 leading-relaxed text-center">
              💡 CLICK empty space in the grid to manually place custom data points!
            </p>
          </div>

          {/* CLASSIFICATION STATS AND METRICS */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3 font-mono">
              Diagnostics Metrics
            </h3>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <div className="bg-slate-950 border border-slate-850 p-3 rounded-xl text-center">
                <span className="text-[10px] font-mono text-slate-400 block">ACCURACY</span>
                <span className="text-lg font-bold font-mono text-emerald-400 mt-1 block">
                  {svmModel ? `${metrics.accuracy.toFixed(1)}%` : '0.0%'}
                </span>
              </div>
              <div className="bg-slate-950 border border-slate-850 p-3 rounded-xl text-center">
                <span className="text-[10px] font-mono text-slate-400 block">SUPPORT VECTS</span>
                <span className="text-lg font-bold font-mono text-indigo-300 mt-1 block">
                  {svmModel ? metrics.supportVectors : '0'}
                </span>
              </div>
              <div className="bg-slate-950 border border-slate-850 p-3 rounded-xl text-center">
                <span className="text-[10px] font-mono text-slate-400 block">F1 SCORE</span>
                <span className="text-lg font-bold font-mono text-amber-400 mt-1 block">
                  {svmModel ? metrics.f1.toFixed(3) : '0.000'}
                </span>
              </div>
            </div>

            {/* CONFUSION MATRIX AND PARAMS SUB-BLOCK */}
            <div className="bg-slate-950/60 border border-slate-850 p-3 rounded-xl grid grid-cols-2 gap-4 items-center">
              <div>
                <span className="text-[10px] text-slate-400 font-mono block mb-2 text-center uppercase tracking-wider">
                  Confusion Matrix
                </span>
                <div className="grid grid-cols-2 gap-1 font-mono text-[9px] w-full max-w-[120px] mx-auto text-center">
                  <div className="bg-emerald-950/40 border border-emerald-900/40 p-1 rounded" title="True Positives (Cats predicted correct)">
                    TP: {svmModel ? metrics.tp : 0}
                  </div>
                  <div className="bg-rose-950/20 border border-rose-900/20 p-1 rounded text-red-400" title="False Dogs (Cats predicted dogs)">
                    FN: {svmModel ? metrics.fn : 0}
                  </div>
                  <div className="bg-rose-950/20 border border-rose-900/20 p-1 rounded text-red-400" title="False Cats (Dogs predicted cats)">
                    FP: {svmModel ? metrics.fp : 0}
                  </div>
                  <div className="bg-blue-950/40 border border-blue-900/40 p-1 rounded" title="True Negatives (Dogs predicted correct)">
                    TN: {svmModel ? metrics.tn : 0}
                  </div>
                </div>
              </div>

              <div className="text-[10px] text-slate-400 space-y-1.5 border-l border-slate-800 pl-4">
                <div className="flex justify-between">
                  <span>Precision:</span>
                  <span className="font-mono text-slate-200">{metrics.precision.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Recall:</span>
                  <span className="font-mono text-slate-200">{metrics.recall.toFixed(3)}</span>
                </div>
                <div className="flex justify-between">
                  <span>Dataset scale:</span>
                  <span className="font-mono text-slate-200">{dataset.length} samples</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* MIDDLE SECTION (COL SPAN 4): SVM OPTIMIZATION CONTROLS */}
        <section className="lg:col-span-4 bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col gap-5 h-full">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Sliders className="w-4.5 h-4.5 text-indigo-400" />
              <h2 className="text-sm font-medium text-slate-200">SVM Hyperparameters</h2>
            </div>

            {/* KERNELS TABS */}
            <div className="bg-slate-950 p-1 border border-slate-800 rounded-xl grid grid-cols-3 mb-4">
              {(['linear', 'rbf', 'polynomial'] as const).map((type) => (
                <button
                  key={type}
                  onClick={() => setKernelType(type)}
                  className={`py-1.5 text-xs font-mono uppercase rounded-lg transition-all cursor-pointer ${
                    kernelType === type
                      ? 'bg-indigo-600 font-semibold text-white shadow-lg'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>

            <div className="space-y-4">
              {/* Regularization C slider */}
              <div>
                <div className="flex justify-between text-xs font-mono text-slate-300 mb-1">
                  <span>Regularization C Penalty</span>
                  <span className="text-indigo-400 font-semibold">{C.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="50.0"
                  step="0.5"
                  value={C}
                  onChange={(e) => setC(parseFloat(e.target.value))}
                  className="w-full accent-indigo-500 bg-slate-950 rounded-lg appearance-none h-1.5 cursor-pointer"
                />
                <span className="text-[9px] text-slate-500 font-sans mt-1 block">
                  Higher values force strict separation (narrow margin). Lower values tolerate mistakes (wider margin).
                </span>
              </div>

              {/* RBF Sigma parameter (conditional on RBF kernel) */}
              {kernelType === 'rbf' && (
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-300 mb-1">
                    <span>RBF Gamma Rad ($\sigma$)</span>
                    <span className="text-sky-400 font-semibold">{rbfSigma.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="1.5"
                    step="0.05"
                    value={rbfSigma}
                    onChange={(e) => setRbfSigma(parseFloat(e.target.value))}
                    className="w-full accent-sky-500 bg-slate-950 rounded-lg appearance-none h-1.5 cursor-pointer"
                  />
                  <span className="text-[9px] text-slate-500 font-sans mt-1 block">
                    Defines spatial radius multiplier. Smaller values yield tighter/spikier contours around vectors.
                  </span>
                </div>
              )}

              {/* Polynomial Degree (conditional) */}
              {kernelType === 'polynomial' && (
                <div>
                  <div className="flex justify-between text-xs font-mono text-slate-300 mb-1">
                    <span>Polynomial Degree (d)</span>
                    <span className="text-amber-400 font-semibold">Degree {polyDegree}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="5"
                    step="1"
                    value={polyDegree}
                    onChange={(e) => setPolyDegree(parseInt(e.target.value))}
                    className="w-full accent-amber-500 bg-slate-950 rounded-lg appearance-none h-1.5 cursor-pointer"
                  />
                  <span className="text-[9px] text-slate-500 font-sans mt-0.5 block">
                    Determines complexity of polynomial lines. Level 1 defaults to linear classification.
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* ACTIVE SMO OPTIMIZATION ENGINE MODULE */}
          <div className="border-t border-slate-800 pt-5 flex-1 flex flex-col justify-between">
            <div>
              <div className="flex items-center gap-1.5 mb-3">
                <Brain className="w-4 h-4 text-emerald-400 animate-pulse" />
                <h3 className="text-xs font-medium text-slate-200">Platt's SMO Optimizer</h3>
              </div>

              {/* Interactive Training Status bar */}
              <div className="bg-slate-950 border border-slate-850 p-3.5 rounded-xl mb-4">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-slate-400">STATUS:</span>
                  <span
                    className={`font-semibold capitalize px-2 py-0.5 rounded text-[10px] font-mono ${
                      trainingStatus === 'converged'
                        ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                        : trainingStatus === 'training'
                          ? 'bg-amber-500/10 text-amber-500 border border-amber-500/30'
                          : 'bg-slate-900 text-slate-400 border border-slate-800'
                    }`}
                  >
                    {trainingStatus === 'untrained' && '● Untrained'}
                    {trainingStatus === 'training' && '⚙️ Running...'}
                    {trainingStatus === 'converged' && '✓ Optimal'}
                    {trainingStatus === 'max_passes' && '⚠ Max Passes'}
                  </span>
                </div>

                <div className="mt-3.5 space-y-1 text-[11px] font-mono text-slate-400">
                  <div className="flex justify-between">
                    <span>Optimization Epoch:</span>
                    <span className="text-slate-200">{currentPass}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Changes in last pass:</span>
                    <span
                      className={`font-semibold font-mono ${
                        alphasChangedInLastPass > 0 ? 'text-amber-400' : 'text-slate-500'
                      }`}
                    >
                      {alphasChangedInLastPass}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span>Bias threshold intercept ($b$):</span>
                    <span className="text-slate-200">{svmModel ? svmModel.b.toFixed(4) : '0.000'}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Active Control Buttons */}
            <div className="grid grid-cols-2 gap-3 mt-4">
              <button
                onClick={handleAutoTrain}
                disabled={trainingStatus === 'training'}
                className="col-span-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 disabled:hover:bg-indigo-600 text-white font-medium text-xs py-2.5 rounded-xl shadow-lg shadow-indigo-600/15 cursor-pointer flex items-center justify-center gap-1.5 font-sans"
              >
                <Play className="w-4 h-4 fill-white" />
                Train Classifier
              </button>
              <button
                onClick={handleStepTrain}
                disabled={trainingStatus === 'training'}
                className="bg-slate-950 hover:bg-slate-900 border border-slate-800 text-slate-300 font-mono text-[11px] py-2 rounded-xl transition-all cursor-pointer flex items-center justify-center gap-1 leading-none"
              >
                <ChevronRight className="w-3.5 h-3.5 text-indigo-400" />
                Single Epoch
              </button>
              <button
                onClick={handleResetModel}
                className="bg-slate-950 hover:bg-slate-900 border border-slate-800 text-slate-300 font-mono text-[11px] py-1.5 rounded-xl transition-all cursor-pointer flex items-center justify-center gap-1 leading-none"
              >
                <RotateCcw className="w-3.5 h-3.5 text-slate-500" />
                Clear Matrix
              </button>
            </div>
          </div>
        </section>

        {/* RIGHT ZONE (COL SPAN 3): DETAILED VISUAL INSPECTOR & CUSTOM UPLOADER */}
        <section className="lg:col-span-3 flex flex-col gap-4">
          <FeatureScanner selectedPoint={selectedPoint} svmModel={svmModel} />

          {/* REAL TIME DRAG & DROP CUSTOM UPLOADER CORE */}
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2.5 font-mono flex items-center gap-1.5">
              <Upload className="w-4 h-4 text-emerald-400" />
              Upload custom image
            </h3>
            
            <div
              onClick={handleUploadClick}
              className="border-2 border-dashed border-slate-800 hover:border-emerald-500/50 rounded-xl p-3 text-center cursor-pointer bg-slate-950 transition-all duration-300 flex flex-col items-center justify-center py-5 group"
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*"
                className="hidden"
              />
              <FileImage className="w-7 h-7 text-slate-600 group-hover:text-emerald-400 group-hover:scale-110 transition-all mb-2" />
              <p className="text-[11px] font-sans text-slate-400 font-medium">Click to select photos</p>
              <span className="text-[9px] text-slate-500 mt-1 block">PNG, JPG, JPEG</span>
            </div>

            {uploading && (
              <div className="mt-3 text-xs text-emerald-400 font-mono flex items-center gap-1.5 bg-emerald-950/20 border border-emerald-900/30 p-2 rounded-lg">
                <SpinnerSmall />
                <span>Running visual convolutions...</span>
              </div>
            )}
          </div>
        </section>
      </main>

      {/* 3. MATHEMATICS AND INFRASTRUCTURE WORKBENCH SECTION */}
      <footer className="max-w-7xl mx-auto mt-6">
        <TheorySection />
        <div className="text-center text-[10px] text-slate-600 font-mono mt-6 border-t border-slate-900 pt-4 pb-2">
          COMPUTED LOCALLY IN CHROMIUM CONTAINER VITE RENDER ENGINE • KAGLE SUBSET DB CACHED.
        </div>
      </footer>

      {/* 4. HIGH FIDELITY POPUP MODAL: CHOOSE CUSTOM UPLOADS CATEGORIES */}
      <AnimatePresence>
        {showLabelModal && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm" id="upload-label-modal">
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="bg-slate-900 border border-slate-800 p-5 rounded-2xl max-w-sm w-full shadow-2xl relative"
            >
              <h3 className="text-sm font-semibold text-slate-100 flex items-center gap-1.5 border-b border-slate-800 pb-2 mb-3">
                <Sparkles className="w-4 h-4 text-emerald-400" />
                Convolutional Feature Report
              </h3>
              
              <div className="flex gap-4 items-center bg-slate-950 p-2 rounded-xl border border-slate-850 mb-4">
                <img
                  src={showLabelModal.url}
                  alt="Raw preview"
                  className="w-14 h-14 object-cover rounded-lg border border-slate-800"
                />
                <div className="font-mono text-xs text-slate-400 space-y-0.5">
                  <div>Warmth: <span className="text-amber-400 font-bold">{showLabelModal.warmth}</span></div>
                  <div>Edges: <span className="text-emerald-400 font-bold">{showLabelModal.edgeDensity}</span></div>
                </div>
              </div>

              <p className="text-xs text-slate-300 leading-relaxed mb-4">
                We successfully calculated the image descriptors. What is the actual class label (+1 or -1) for this image?
              </p>

              <div className="grid grid-cols-2 gap-3 mb-2">
                <button
                  onClick={() => handleConfirmCustomUpload('cat')}
                  className="bg-amber-600 hover:bg-amber-500 font-medium text-xs py-2 rounded-xl text-white cursor-pointer transition-all text-center"
                >
                  It is a Cat (+1)
                </button>
                <button
                  onClick={() => handleConfirmCustomUpload('dog')}
                  className="bg-blue-600 hover:bg-blue-500 font-medium text-xs py-2 rounded-xl text-white cursor-pointer transition-all text-center"
                >
                  It is a Dog (-1)
                </button>
              </div>

              <button
                onClick={() => setShowLabelModal(null)}
                className="w-full mt-2 font-mono text-[10px] text-slate-500 hover:text-slate-300 text-center leading-normal"
              >
                Abort Upload
              </button>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SpinnerSmall() {
  return (
    <svg className="animate-spin h-3.5 w-3.5 text-emerald-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
  );
}
