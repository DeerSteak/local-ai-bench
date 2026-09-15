import { useContext } from "react";
import { ExternalServerContext } from "./ExternalServerContext";

export function ExternalServerNote() {
  const notes = useContext(ExternalServerContext);
  return notes.map(note => <p key={note} style={{ margin: "12px 16px", color: "#57606a", fontSize: 12 }}>{note}</p>);
}
