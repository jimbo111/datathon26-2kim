import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 30000,
});

// --- Data ---
export const listDatasets = () => api.get("/data/datasets").then((r) => r.data);
export const loadDataset = (filename, subdir = "raw") =>
  api.post(`/data/load/${filename}?subdir=${subdir}`).then((r) => r.data);
export const getDataInfo = () => api.get("/data/info").then((r) => r.data);
export const getDataHead = (n = 20) =>
  api.get("/data/head", { params: { n } }).then((r) => r.data);
export const getColumns = () => api.get("/data/columns").then((r) => r.data);
export const getStats = () => api.get("/data/stats").then((r) => r.data);

// --- Charts ---
export const getDistribution = (column, bins = 30) =>
  api.get("/charts/distribution", { params: { column, bins } }).then((r) => r.data);
export const getCorrelation = (method = "pearson") =>
  api.get("/charts/correlation", { params: { method } }).then((r) => r.data);
export const getBarChart = (column, topN = 20) =>
  api.get("/charts/bar", { params: { column, top_n: topN } }).then((r) => r.data);
export const getScatter = (x, y, hue = null) =>
  api.get("/charts/scatter", { params: { x, y, hue } }).then((r) => r.data);
export const getTimeSeries = (dateCol, valueCol, freq = null) =>
  api.get("/charts/timeseries", { params: { date_col: dateCol, value_col: valueCol, freq } }).then((r) => r.data);

export default api;
