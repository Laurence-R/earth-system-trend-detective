import { OrbitControls } from "@react-three/drei";
import { Canvas, type CanvasProps } from "@react-three/fiber";
import { Suspense, useState } from "react";
import { WebGPURenderer } from "three/webgpu";
import { GlassParticleLayer } from "../layers/GlassParticleLayer";
import { layerRegistry, type LayerDefinition } from "../layers/registry";
import { mockNormalized } from "../mock/snapshot";
import { Earth } from "./Earth";
import { FlyCamera } from "./FlyCamera";
import { Markers } from "./Markers";
import { rendererKindFromBackend, type RendererKind } from "./rendererLabel";

type CreateGl = NonNullable<CanvasProps["gl"]>;

export type SceneLayer = {
  id: string;
  visible: boolean;
  opacity: number;
};

type EarthCanvasProps = {
  forceWebGL: boolean;
  flying: boolean;
  layers: SceneLayer[];
  locationId: string;
  onKind: (kind: RendererKind) => void;
  onFlyDone: () => void;
  onSelectLocation: (id: string) => void;
};

function Scene({
  flying,
  layers,
  locationId,
  onFlyDone,
  onSelectLocation,
}: {
  flying: boolean;
  layers: SceneLayer[];
  locationId: string;
  onFlyDone: () => void;
  onSelectLocation: (id: string) => void;
}) {
  const [controlEpoch, setControlEpoch] = useState(0);

  return (
    <>
      <color attach="background" args={["#0b1220"]} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[4, 2, 3]} intensity={1.1} />
      <Suspense fallback={null}>
        <Earth />
      </Suspense>
      <Markers selectedId={locationId} onSelect={onSelectLocation} />
      {layerRegistry.map((layer) => (
        <LayerView
          key={layer.id}
          layer={layer}
          control={layers.find((item) => item.id === layer.id)}
          locationId={locationId}
        />
      ))}
      <FlyCamera
        flying={flying}
        onDone={() => {
          setControlEpoch((epoch) => epoch + 1);
          onFlyDone();
        }}
      />
      <OrbitControls key={controlEpoch} enablePan={false} enabled={!flying} />
    </>
  );
}

function LayerView({
  layer,
  control,
  locationId,
}: {
  layer: LayerDefinition;
  control: SceneLayer | undefined;
  locationId: string;
}) {
  if (!control) return null;
  return (
    <GlassParticleLayer
      layer={layer}
      visible={control.visible}
      opacity={control.opacity}
      intensity={mockNormalized(layer.id, locationId)}
    />
  );
}

function createGl(
  forceWebGL: boolean,
  onKind: (kind: RendererKind) => void,
): CreateGl {
  return async (props) => {
    const renderer = new WebGPURenderer({
      canvas: props.canvas as HTMLCanvasElement,
      antialias: true,
      forceWebGL,
    });
    await renderer.init();
    onKind(
      rendererKindFromBackend(
        renderer.backend as { isWebGPUBackend?: boolean },
      ),
    );
    return renderer;
  };
}

export function EarthCanvas({
  forceWebGL,
  flying,
  layers,
  locationId,
  onKind,
  onFlyDone,
  onSelectLocation,
}: EarthCanvasProps) {
  return (
    <Canvas
      key={forceWebGL ? "webgl" : "auto"}
      camera={{ position: [0, 0, 3.2], fov: 45 }}
      dpr={[1, 2]}
      gl={createGl(forceWebGL, onKind)}
    >
      <Scene
        flying={flying}
        layers={layers}
        locationId={locationId}
        onFlyDone={onFlyDone}
        onSelectLocation={onSelectLocation}
      />
    </Canvas>
  );
}
