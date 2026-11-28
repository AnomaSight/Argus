type Props = { input: string; heatmap: string; range: [number, number] };

// Draws the heatmap over the model's input view, never over the raw upload (C5).
export default function HeatmapOverlay({ input, heatmap, range }: Props) {
  return (
    <div>
      <div className="overlay">
        <img src={`data:image/png;base64,${input}`} alt="The image as the model sees it" />
        <img src={`data:image/png;base64,${heatmap}`} alt="Anomaly heatmap" />
      </div>
      <div className="legend" aria-label={`Heatmap scale from ${range[0]} to ${range[1]}`}>
        <span>{range[0].toFixed(2)}</span>
        <span className="legend-bar" />
        <span>{range[1].toFixed(2)}</span>
      </div>
    </div>
  );
}
