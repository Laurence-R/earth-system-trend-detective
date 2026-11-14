import { useState } from "react";
import { EarthCanvas } from "./scene/EarthCanvas";
import { rendererLabel, type RendererKind } from "./scene/rendererLabel";

export function App() {
  const [forceWebGL, setForceWebGL] = useState(false);
  const [kind, setKind] = useState<RendererKind>("webgl");

  return (
    <div className="app">
      <EarthCanvas forceWebGL={forceWebGL} onKind={setKind} />
      <div className="hud">
        <p className="badge">Renderer: {rendererLabel(kind)}</p>
        <button type="button" onClick={() => setForceWebGL((value) => !value)}>
          {forceWebGL ? "Use automatic renderer" : "Force WebGL"}
        </button>
      </div>
    </div>
  );
}
