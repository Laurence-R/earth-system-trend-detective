import { useState } from "react";
import { LayerPanel, type LayerControl } from "./layers/LayerPanel";
import { initialLayerVisibility, layerRegistry } from "./layers/registry";
import { mockLocations } from "./mock/snapshot";
import { EarthCanvas } from "./scene/EarthCanvas";
import { FpsReadout } from "./scene/FpsReadout";
import { rendererLabel, type RendererKind } from "./scene/rendererLabel";

const initialControls: LayerControl[] = layerRegistry.map((layer) => ({
  id: layer.id,
  visible: initialLayerVisibility(layer.id),
  opacity: 0.75,
}));

export function App() {
  const [forceWebGL, setForceWebGL] = useState(false);
  const [flying, setFlying] = useState(false);
  const [kind, setKind] = useState<RendererKind>("webgl");
  const [locationId, setLocationId] = useState("taichung");
  const [controls, setControls] = useState(initialControls);
  const location = mockLocations.find((item) => item.id === locationId);

  return (
    <div className="app">
      <EarthCanvas
        forceWebGL={forceWebGL}
        flying={flying}
        layers={controls}
        locationId={locationId}
        onKind={setKind}
        onFlyDone={() => setFlying(false)}
        onSelectLocation={setLocationId}
      />
      <div className="hud">
        <p className="badge">Renderer: {rendererLabel(kind)}</p>
        <FpsReadout />
        <button type="button" onClick={() => setForceWebGL((value) => !value)}>
          {forceWebGL ? "Use automatic renderer" : "Force WebGL"}
        </button>
        <button type="button" onClick={() => setFlying(true)} disabled={flying}>
          Fly to Taiwan
        </button>
        <a href={`${import.meta.env.BASE_URL}lab/`}>Form lab</a>
      </div>
      <LayerPanel
        layers={layerRegistry}
        controls={controls}
        locationName={location?.nameEn ?? locationId}
        onToggle={(id) =>
          setControls((current) =>
            current.map((layer) =>
              layer.id === id ? { ...layer, visible: !layer.visible } : layer,
            ),
          )
        }
        onOpacity={(id, opacity) =>
          setControls((current) =>
            current.map((layer) =>
              layer.id === id ? { ...layer, opacity } : layer,
            ),
          )
        }
      />
      <p className="credit">
        Surface: Blue Marble,{" "}
        <a href="https://visibleearth.nasa.gov/images/57730/blue-marble-land-surface-ocean-color-sea-ice-and-clouds">
          NASA Visible Earth
        </a>
        . Closer Taiwan view: Blue Marble shaded relief,{" "}
        <a href="https://gibs.earthdata.nasa.gov/">NASA GIBS</a>. NASA insignia
        is not used.
      </p>
    </div>
  );
}
