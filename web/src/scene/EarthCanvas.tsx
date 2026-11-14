import { OrbitControls } from "@react-three/drei";
import { Canvas, type CanvasProps } from "@react-three/fiber";
import { WebGPURenderer } from "three/webgpu";
import { rendererKindFromBackend, type RendererKind } from "./rendererLabel";

type CreateGl = NonNullable<CanvasProps["gl"]>;

type EarthCanvasProps = {
  forceWebGL: boolean;
  onKind: (kind: RendererKind) => void;
};

function Scene() {
  return (
    <>
      <color attach="background" args={["#0b1220"]} />
      <ambientLight intensity={0.55} />
      <directionalLight position={[4, 2, 3]} intensity={1.3} />
      <mesh>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          color="#7aa2c4"
          roughness={0.65}
          metalness={0.05}
        />
      </mesh>
      <OrbitControls enablePan={false} />
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

export function EarthCanvas({ forceWebGL, onKind }: EarthCanvasProps) {
  return (
    <Canvas
      key={forceWebGL ? "webgl" : "auto"}
      camera={{ position: [0, 0, 3.2], fov: 45 }}
      dpr={[1, 2]}
      gl={createGl(forceWebGL, onKind)}
    >
      <Scene />
    </Canvas>
  );
}
