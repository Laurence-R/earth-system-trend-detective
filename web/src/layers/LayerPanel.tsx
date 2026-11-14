import type { LayerDefinition } from "./registry";

export type LayerControl = {
  id: string;
  visible: boolean;
  opacity: number;
};

type LayerPanelProps = {
  layers: LayerDefinition[];
  controls: LayerControl[];
  locationName: string;
  onToggle: (id: string) => void;
  onOpacity: (id: string, opacity: number) => void;
};

export function LayerPanel({
  layers,
  controls,
  locationName,
  onToggle,
  onOpacity,
}: LayerPanelProps) {
  return (
    <aside className="layers">
      <p className="layers-title">Layers</p>
      <p className="layers-note">Values at {locationName}</p>
      <ul>
        {layers.map((layer) => {
          const control = controls.find((item) => item.id === layer.id);
          if (!control) return null;
          return (
            <li key={layer.id}>
              <label>
                <input
                  type="checkbox"
                  checked={control.visible}
                  onChange={() => onToggle(layer.id)}
                />
                <span>{layer.nameEn}</span>
              </label>
              <input
                type="range"
                min={0.15}
                max={1}
                step={0.05}
                value={control.opacity}
                aria-label={`${layer.nameEn} opacity`}
                onChange={(event) =>
                  onOpacity(layer.id, Number(event.target.value))
                }
              />
            </li>
          );
        })}
      </ul>
    </aside>
  );
}
