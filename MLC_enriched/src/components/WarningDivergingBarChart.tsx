import "../style/warningDivergingBarChart.css";
import { useEffect, useState } from "react";
import { fetchShapValues, fetchPrediction } from "../utils/api_getPrediction";
import { logUserAction } from "../utils/logger";
import { Bar } from 'react-chartjs-2';
import ChartDataLabels from 'chartjs-plugin-datalabels';
import Loader from "./Loader";
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Title,
  type Align,
} from 'chart.js'; 

ChartJS.register(BarElement, CategoryScale, LinearScale, Tooltip, Title, ChartDataLabels);

type ShapFeature = {
  feature: string;
  feature_value: string | number;
  shap_value: number;
  abs_val: number;
};

type Props = {
  selectedPatientId: string;
};

const WarningShapChart = ({ selectedPatientId }: Props) => {
  const [activeTab, setActiveTab] = useState("chart1");
  const [shapData, setShapData] = useState<ShapFeature[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [probability, setProbability] = useState<number | null>(null);
  const COLOR_RED = '#e5383b';
  const COLOR_GREEN = '#6e9887';

  const handleTabChange = (tabName: string) => {
    setActiveTab(tabName);

    logUserAction("SHAP_Tab_Change", {
      selectedTab: tabName,
      patientId: selectedPatientId,
    });
  };

  const ErrorMessage = ({ message }: { message: string }) => (
    <div className="error-message">
      <h3>No SHAP explanation available</h3>
      <p>{message}</p>
      <p>Please check if the prediction data is available for this patient.</p>
    </div>
  );

  useEffect(() => {
    setError(null);
    setShapData(null);
    setLoading(true);

    const token = localStorage.getItem("token");
    fetchShapValues(selectedPatientId, token)
      .then(({ shap_values, error }) => {
        setShapData(shap_values);
        setError(error);
        setLoading(false);
      });
  }, [selectedPatientId]);

  useEffect(() => {
    if (!selectedPatientId) return;
    const token = localStorage.getItem("token");
    fetchPrediction(selectedPatientId, token)
      .then(({ prediction_proba }) => {
        if (prediction_proba !== undefined && prediction_proba !== null) {
          setProbability(prediction_proba);
        } else {
          setProbability(null);
        }
      })
      .catch(() => setProbability(null));
  }, [selectedPatientId]);

  if (loading) return <Loader />;
  if (error || !shapData) return <ErrorMessage message={"The necessaries data are not available for the SHAP explanation."} />;

  const features = Array.isArray(shapData)
    ? shapData.map(item => ({
      label: item.feature,
      value: item.shap_value,
      meta: `${item.feature}: ${item.feature_value}`
    }))
    : [];

  const sortedFeatures = [...features].sort(
    (a, b) => Math.abs(b.value) - Math.abs(a.value)
  );

  const data = {
    labels: sortedFeatures.map((f) => f.label),
    datasets: [
      {
        label: 'SHAP Impact',
        data: sortedFeatures.map((f) => f.value),
        backgroundColor: (context: any) => {
          const value = context.dataset.data[context.dataIndex];
          return value >= 0 ? COLOR_GREEN : COLOR_RED;
        },
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    indexAxis: 'y' as const,
    scales: {
      x: {
        grace: '5%',
        title: {
          display: false,
        },
        ticks: {
          color: 'black',
        },
        grid: {
          color: (ctx: any) => (ctx.tick.value === 0 ? COLOR_GREEN : '#939480'),
          lineWidth: (ctx: any) => (ctx.tick.value === 0 ? 2 : 1),
        },
      },
      y: {
        title: {
          display: false,
        },
        ticks: {
          color: (context: any) => {
            const label = context.tick.label;
            return label === 'Body Temperature' ? COLOR_RED : 'black';
          },
          font: (ctx: any) => {
            const label = ctx.tick.label;
            return {
              weight: label === 'Body Temperature' ? 'bold' as const : 'normal' as const,
            };
          },
        }
      },
    },
    elements: {
      bar: {
        hoverBackgroundColor: '#222',
        hoverBorderColor: '#000',
      },
    },
    plugins: {
      legend: {
        display: true,
        position: 'bottom' as const,
        labels: {
          color: 'black',
          font: {
            size: 12,
            weight: 'bold' as const,
          },
          boxWidth: 20,
          padding: 15,
        },
      },
      title: {
        display: true,
        text: 'Impact of Medical Factors on the AI\'s Prediction',
        color: '#222',
        font: {
          size: 18,
          weight: 'bold' as const,
        },
        padding: {
          top: 20,
          bottom: 20,
        },
      },
      datalabels: {
        color: '#f0f0f0',
        clamp: true,
        formatter: (value: number) => {
          if (Math.abs(value) < 0.08) return '';
          return value >= 0 ? `+${value.toFixed(2)}` : value.toFixed(2);
        },
        anchor: 'center' as Align,
        align: 'center' as Align,
      },
      tooltip: {
        callbacks: {
          title: (context: any) => {
            const index = context[0].dataIndex;
            const feature = sortedFeatures[index];

            const rawValue = feature.meta.split(':')[1]?.trim();
            const roundedValue = rawValue ? parseFloat(rawValue).toFixed(2) : '';

            return `${feature.label}: ${roundedValue}`;
          },
          label: (context: any) => {
            const val = context.parsed.x;

            const isPositive = val >= 0;
            const contributionText = isPositive
              ? 'That reduces the predicted risk'
              : 'That increases the predicted risk';

            const causalityWarning = 'SHAP explains the model\'s behavior, not causality.';
            const individualScopeWarning = 'This explanation is specific to this patient and may not apply to others.';
            const interpretationWarning = '⚠️ High or low SHAP values do not mean the feature value itself is high or low, or that changing it will improve the patient\'s condition.';

            return [
              `SHAP Value: ${val >= 0 ? '+' : ''}${val.toFixed(2)}`,
              contributionText,
              interpretationWarning,
              causalityWarning,
              individualScopeWarning
            ];
          },
        },
        titleFont: {
          size: 16,
        },
        bodyFont: {
          size: 12
        },
      },
    },
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case "chart1":
        return (
          <div className="tab-1-content">
            <h3 className="instructions-title">🧠 What is a SHAP plot?</h3>
            <p className="instructions-note">
              It shows the most influential factiors on this specific prediction.
            </p>

            <h3 className="instructions-title">🔍 How to read it</h3>
            <ul className="no-dot">
              <li>📊 <strong>Each bar</strong> an influential factor on the prediction</li>
              <li>➡️ <strong>Red Bar to the right</strong> = <strong>increases the risk of death</strong> (harmful effect)</li>
              <li>⬅️ <strong>Green Bar to the left</strong> = <strong>reduces the risk</strong> (protective effect)</li>
            </ul>

            <p className="info-note">
              📉 The longer the bar, the stronger the impact on the model’s prediction.
            </p>

            <h3 className="instructions-title">⚠️ Keep in mind</h3>
            <ul className="instructions-note">
              <li>The explanation does not prove what caused the outcome.</li>
              <li>This is not diagnosis, just an explanation of the model.</li>
              <li>This only explains one individual case</li>
              <li>This chart is adapted for readability. Directions and colors reflect the model's medical risk interpretation.</li>
            </ul>
          </div>
        );
      case "chart2":
        return (
          <div className="tab-2-content">
            <div className="bar">
              <Bar data={data} options={options} />
            </div>
            <div className="custom-legend">
              <div><span style={{ backgroundColor: COLOR_RED }} className="legend-color"></span> Negative Contribution</div>
              <div><span style={{ backgroundColor: COLOR_GREEN }} className="legend-color"></span> Positive Contribution</div>
            </div>
          </div>
        );
      case "chart3":
        return (
          <div className="tab-3-content">
            <h3>⚠️ Important Reminder</h3>
            <p>
              SHAP helps explain a prediction, it doesn't diagnose or prove the cause.
            </p>

            <h4>🚫 What SHAP doesn’t tell you:</h4>
            <ul>
              <li><strong>No causality:</strong> High impact ≠ the cause of the prediction</li>
              <li><strong>No diagnosis:</strong> It explains the model, not the patient</li>
              <li><strong>Not universal:</strong> It only applies to this individual case</li>
            </ul>

            <h4>✅ Good Practices</h4>
            <ul>
              <li>🔍 Cross-check with clinical info</li>
              <li>⚖️ Be aware of model biases</li>
              <li>❓ Investigate surprising results</li>
              <li>📝 Use explanations to support—not replace—your reasoning</li>
            </ul>

            <p className="post-note">
              🧠 SHAP is a tool, not a decision-maker. Always apply clinical judgement.
            </p>

            <p className="download-note">
              📥 Want more details? ‎ <a href="../../public/Understanding_SHAP_Values.pdf" download>Download the SHAP interpretation guide (PDF)</a>
            </p>
          </div>
        );
    }
  }


  return (
    <div className="warning-chart-container">
      <h2 className="warning-bar-chart-title">Key Medical Factors Influencing the AI Prediction</h2>

      <div className="tabs">
        <button
          className={activeTab === "chart1" ? "tab-active" : "tab"}
          onClick={() => handleTabChange("chart1")}
        >
          Instructions
        </button>
        <button
          className={activeTab === "chart2" ? "tab-active" : "tab"}
          onClick={() => handleTabChange("chart2")}
        >
          Chart
        </button>
        <button
          className={activeTab === "chart3" ? "tab-active" : "tab"}
          onClick={() => handleTabChange("chart3")}
        >
          More Info
        </button>
      </div>

      <div className="tab-content">
        {renderTabContent()}
      </div>

      <div className="probability">
        <h3>Predicted Risk of Death in the next 30 days</h3>
        {probability !== null ? (
          <p>
            {Math.round((1 - probability) * 100)}%
          </p>
        ) : (
          <p>Loading...</p>
        )}
      </div>
    </div>
  );
};

export default WarningShapChart;
