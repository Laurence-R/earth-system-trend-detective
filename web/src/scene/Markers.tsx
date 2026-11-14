import { latLonToVector3 } from "../geo/latLon";
import { mockLocations } from "../mock/snapshot";

export function Markers() {
  return (
    <>
      {mockLocations.map((location) => {
        const position = latLonToVector3(location.lat, location.lon, 1.008);
        return (
          <mesh
            key={location.id}
            position={[position.x, position.y, position.z]}
          >
            <sphereGeometry args={[0.0045, 12, 12]} />
            <meshBasicMaterial color="#ffd166" />
          </mesh>
        );
      })}
    </>
  );
}
