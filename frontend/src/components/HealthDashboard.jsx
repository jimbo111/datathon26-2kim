import { useState, useCallback } from "react";
import Chart from "./Chart";
import {
  getChoroplethFoodAccess,
  getChoroplethDiabetes,
  getChoroplethObesity,
  getChoroplethLifeExpectancy,
  getScatterFoodVsDiabetes,
  getScatterFoodVsObesity,
  getBarLifeExpectancyIncome,
} from "../api/client";
import HealthSummaryTable from "./HealthSummaryTable";

const TABS = [
  { id: "maps", label: "Choropleth Maps" },
  { id: "scatter", label: "Correlations" },
  { id: "equity", label: "Health Equity" },
];

export default function HealthDashboard() {
  const [activeTab, setActiveTab] = useState("maps");

  return (
    <div className="dashboard">
      <header>
        <h1>Your Zip Code is Your Health Sentence</h1>
        <p className="subtitle">
          Exploring how geography, food access, and income shape health outcomes
          across the United States
        </p>
      </header>

      {/* Tab navigation */}
      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab-btn ${activeTab === tab.id ? "active" : ""}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === "maps" && <MapsTab />}
      {activeTab === "scatter" && <ScatterTab />}
      {activeTab === "equity" && <EquityTab />}
    </div>
  );
}

function MapsTab() {
  const fetchFoodAccess = useCallback(() => getChoroplethFoodAccess(), []);
  const fetchDiabetes = useCallback(() => getChoroplethDiabetes(), []);
  const fetchObesity = useCallback(() => getChoroplethObesity(), []);
  const fetchLifeExp = useCallback(() => getChoroplethLifeExpectancy(), []);

  return (
    <>
      {/* Side-by-side choropleths */}
      <section className="panel">
        <h2>Side-by-Side: Food Access vs Disease</h2>
        <p className="panel-desc">
          States with lower food access scores (red) cluster with higher diabetes
          rates — food deserts and chronic disease are not randomly distributed.
        </p>
        <div className="chart-grid two-col">
          <Chart
            fetchFn={fetchFoodAccess}
            title="Food Access Score"
            style={{ height: 450 }}
          />
          <Chart
            fetchFn={fetchDiabetes}
            title="Diabetes Rate"
            style={{ height: 450 }}
          />
        </div>
      </section>

      {/* Obesity + Life Expectancy */}
      <section className="panel">
        <h2>Obesity and Life Expectancy</h2>
        <p className="panel-desc">
          The obesity map mirrors diabetes patterns. Life expectancy reveals the
          ultimate cost — up to 10 years of life lost between the best and worst
          states.
        </p>
        <div className="chart-grid two-col">
          <Chart
            fetchFn={fetchObesity}
            title="Obesity Rate"
            style={{ height: 450 }}
          />
          <Chart
            fetchFn={fetchLifeExp}
            title="Life Expectancy"
            style={{ height: 450 }}
          />
        </div>
      </section>
    </>
  );
}

function ScatterTab() {
  const fetchFoodDiabetes = useCallback(() => getScatterFoodVsDiabetes(), []);
  const fetchFoodObesity = useCallback(() => getScatterFoodVsObesity(), []);

  return (
    <>
      <section className="panel">
        <h2>Food Access vs Chronic Disease</h2>
        <p className="panel-desc">
          Each dot is a state. Size = median income, color = % minority
          population. The trend line confirms: lower food access scores correlate
          with higher disease rates, even when accounting for income.
        </p>
        <div className="chart-grid two-col">
          <Chart
            fetchFn={fetchFoodDiabetes}
            title="Food Access vs Diabetes"
            style={{ height: 450 }}
          />
          <Chart
            fetchFn={fetchFoodObesity}
            title="Food Access vs Obesity"
            style={{ height: 450 }}
          />
        </div>
      </section>
    </>
  );
}

function EquityTab() {
  const fetchLifeExpIncome = useCallback(() => getBarLifeExpectancyIncome(), []);

  return (
    <>
      <section className="panel">
        <h2>Life Expectancy by Income Quintile</h2>
        <p className="panel-desc">
          States in the lowest income quintile live an average of 4+ fewer years
          than the highest. The gap is a direct measure of structural inequality.
        </p>
        <div className="chart-grid">
          <Chart
            fetchFn={fetchLifeExpIncome}
            title="Life Expectancy by Income"
            style={{ height: 400 }}
          />
        </div>
      </section>

      <section className="panel">
        <h2>Most Disadvantaged States</h2>
        <p className="panel-desc">
          Bottom 10 states by food access score — these are the places where zip
          code most constrains health outcomes.
        </p>
        <HealthSummaryTable />
      </section>
    </>
  );
}
