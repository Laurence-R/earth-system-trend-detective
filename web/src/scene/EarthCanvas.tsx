import { OrbitControls } from "@react-three/drei";
import { Canvas, type CanvasProps } from "@react-three/fiber";
import { Suspense, useState } from "react";
import { WebGPURenderer } from "three/webgpu";
import { Earth } from "./Earth";
import { FlyCamera } from "./FlyCamera";
import { Markers } from "./Markers";
import { rendererKindFromBackend, type RendererKind } from "./rendererLabel";

type CreateGl = NonNullable<CanvasProps["gl"]>;

type EarthCanvasProps = {
  forceWebGL: boolean;
  flying: boolean;
  onKind: (kind: RendererKind) => void;
  onFlyDone: () => void;
};

function Scene({
  flying,
  onFlyDone,
}: {
  flying: boolean;
  onFlyDone: () => void;
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
      <Markers />
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
  onKind,
  onFlyDone,
}: EarthCanvasProps) {
  return (
    <Canvas
      key={forceWebGL ? "webgl" : "auto"}
      camera={{ position: [0, 0, 3.2], fov: 45 }}
      dpr={[1, 2]}
      gl={createGl(forceWebGL, onKind)}
    >
      <Scene flying={flying} onFlyDone={onFlyDone} />
    </Canvas>
  );
}
