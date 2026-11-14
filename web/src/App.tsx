import { useState } from "react";
import { EarthCanvas } from "./scene/EarthCanvas";
import { rendererLabel, type RendererKind } from "./scene/rendererLabel";

export function App() {
  const [forceWebGL, setForceWebGL] = useState(false);
  const [flying, setFlying] = useState(false);
  const [kind, setKind] = useState<RendererKind>("webgl");

  return (
    <div className="app">
      <EarthCanvas
        forceWebGL={forceWebGL}
        flying={flying}
        onKind={setKind}
        onFlyDone={() => setFlying(false)}
      />
      <div className="hud">
        <p className="badge">Renderer: {rendererLabel(kind)}</p>
        <button type="button" onClick={() => setForceWebGL((value) => !value)}>
          {forceWebGL ? "Use automatic renderer" : "Force WebGL"}
        </button>
        <button type="button" onClick={() => setFlying(true)} disabled={flying}>
          Fly to Taiwan
        </button>
      </div>
      <p className="credit">
        Surface: Blue Marble,{" "}
        <a href="https://visibleearth.nasa.gov/images/57730/blue-marble-land-surface-ocean-color-sea-ice-and-clouds">
          NASA Visible Earth
        </a>
        . NASA insignia is not used.
      </p>
    </div>
  );
}
